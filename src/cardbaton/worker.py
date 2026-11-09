"""Standard-library worker sent to the remote host through OpenSSH."""

import csv
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ACTIVE = {"starting", "running", "pausing", "cancelling"}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


@contextmanager
def lease(path, *, blocking=False):
    with path.open("a") as file:
        mode = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(file, mode)
        except BlockingIOError:
            yield None
        else:
            # Close releases this open-file-description lock once the last
            # inherited descriptor closes. LOCK_UN would release it early.
            yield file.fileno()


def smi(query, *fields):
    command = ["nvidia-smi", f"--query-{query}={','.join(fields)}", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True, timeout=8)
    return [[item.strip() for item in row] for row in csv.reader(output.splitlines())]


def inventory(root, idle_mib):
    active = {row[0] for row in smi("compute-apps", "gpu_uuid", "pid") if row}
    devices = []
    for index, uuid, used in smi("gpu", "index", "uuid", "memory.used"):
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
            raise ValueError(f"Unrecognized physical GPU UUID {uuid!r}")
        with lease(root / f"gpu-{uuid}.lock") as fd:
            devices.append(dict(index=int(index), uuid=uuid, memory_used_mib=int(used),
                                available=fd is not None and int(used) <= idle_mib and uuid not in active))
    return devices


def status(directory):
    current = read(directory / "status.json")
    if current and current["state"] in ACTIVE:
        with lease(directory / "attempt.lock") as fd:
            if fd is not None:
                current = read(directory / "status.json")
                if current and current["state"] in ACTIVE:
                    current.update(state="lost", detail="The worker exited without a final status.")
                    write(directory / "status.json", current)
    return current


def start(root, directory, request):
    from contextlib import ExitStack

    directory.mkdir(mode=0o700, exist_ok=True)
    current = status(directory)
    if current:
        return current
    with lease(directory / "attempt.lock") as attempt_fd:
        if attempt_fd is None:
            return dict(state="starting")
        current = read(directory / "status.json")
        if current:
            return current
        devices = {item["index"]: item for item in inventory(root, request["idle_mib"])}
        indices = request["gpus"]
        if any(not devices.get(index, {}).get("available") for index in indices):
            return dict(state="busy")
        uuids = [devices[index]["uuid"] for index in indices]
        with ExitStack() as stack:
            gpu_fds = [stack.enter_context(lease(root / f"gpu-{uuid}.lock")) for uuid in sorted(uuids)]
            if None in gpu_fds:
                return dict(state="busy")
            # Recheck external users after acquiring our leases.
            active = {row[0] for row in smi("compute-apps", "gpu_uuid", "pid") if row}
            memory = {uuid: int(used) for uuid, used in smi("gpu", "uuid", "memory.used")}
            if any(uuid in active or memory.get(uuid, float("inf")) > request["idle_mib"] for uuid in uuids):
                return dict(state="busy")
            spec = request["spec"] | {"gpu_uuids": uuids, "gpu_fds": gpu_fds}
            write(directory / "spec.json", spec)
            worker = directory / "worker.py"
            worker.write_text(request["worker"])
            initial = dict(state="starting", gpus=indices, started_at=time.time())
            write(directory / "status.json", initial)
            try:
                with (directory / "worker.log").open("ab") as log:
                    subprocess.Popen([sys.executable, str(worker), "run", str(directory)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, pass_fds=(attempt_fd, *gpu_fds))
            except OSError as exc:
                initial.update(state="failed", detail=str(exc))
                write(directory / "status.json", initial)
            return read(directory / "status.json")


def signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def stop(process):
    signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    signal_group(process, signal.SIGKILL)
    process.wait()


def run(directory):
    spec = read(directory / "spec.json")
    current = read(directory / "status.json")
    process = None
    cwd = Path(spec.get("cwd") or Path.home()).expanduser().resolve()
    pause_file = directory / "pause.request"
    ready_file = directory / "checkpoint.json"
    environment = os.environ | spec.get("env", {})
    environment.pop("SSHGPU_CHECKPOINT", None)
    environment.update(CUDA_VISIBLE_DEVICES=",".join(spec["gpu_uuids"]),
                       SSHGPU_GPU_INDICES=",".join(str(index) for index in current["gpus"]),
                       SSHGPU_ATTEMPT_ID=directory.name,
                       SSHGPU_ATTEMPT_DIR=str(directory.resolve()),
                       SSHGPU_JOB_ID=spec["job_id"], SSHGPU_PAUSE_FILE=str(pause_file),
                       SSHGPU_CHECKPOINT_READY=str(ready_file))
    if spec.get("checkpoint"):
        environment["SSHGPU_CHECKPOINT"] = spec["checkpoint"]
    started = time.monotonic()
    timeout = spec.get("timeout_seconds")
    deadline = started + timeout if timeout is not None else float("inf")
    grace = spec.get("checkpoint_grace_seconds", 60)
    resumable = bool(spec.get("resume_command"))
    stop_reason = None
    pause_requested = False
    try:
        with (directory / "job.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(spec["argv"], cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True,
                                       pass_fds=tuple(spec["gpu_fds"]))
            current.update(state="running", pid=process.pid)
            write(directory / "status.json", current)
            while process.poll() is None:
                command = read(directory / "request.json", {}).get("action")
                expired = time.monotonic() >= deadline
                if command == "cancel" or (expired and (not resumable or time.monotonic() >= deadline + grace)):
                    stop_reason = "cancelled" if command == "cancel" else "timed_out"
                    current["state"] = "cancelling"
                    write(directory / "status.json", current)
                    stop(process)
                    break
                if resumable and (command == "pause" or expired) and not pause_requested:
                    pause_file.touch()
                    pause_requested = True
                    current["state"] = "pausing"
                    write(directory / "status.json", current)
                time.sleep(0.2)
            # A foreground command may have left children in its process group.
            signal_group(process, signal.SIGKILL)
        current["exit_code"] = process.returncode
        if stop_reason:
            current["state"] = stop_reason
        elif process.returncode == 75 and pause_requested:
            marker = read(ready_file, {})
            checkpoint = marker.get("checkpoint") if isinstance(marker, dict) else None
            if isinstance(checkpoint, str) and checkpoint and (cwd / checkpoint).exists():
                current.update(state="paused", checkpoint=str((cwd / checkpoint).resolve()))
            else:
                current.update(state="failed", detail="Checkpoint exit without a valid checkpoint marker.")
        else:
            current["state"] = "succeeded" if process.returncode == 0 else "failed"
    except (OSError, ValueError, TypeError) as exc:
        if process is not None and process.poll() is None:
            stop(process)
        current.update(state="failed", detail=str(exc))
    finally:
        current["finished_at"] = time.time()
        write(directory / "status.json", current)


def rpc(request):
    root = Path(request["root"]).expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    action = request["action"]
    if action == "probe":
        return inventory(root, request["idle_mib"])
    if action == "statuses":
        return {attempt: status(attempt_dir(root, attempt)) for attempt in request["attempts"]}
    directory = attempt_dir(root, request["attempt"])
    if action == "start":
        return start(root, directory, request)
    if action == "logs":
        log = directory / "job.log"
        if not log.exists():
            log = directory / "worker.log"
        if not log.exists():
            return ""
        with log.open("rb") as stream:
            stream.seek(max(0, stream.seek(0, os.SEEK_END) - request["max_bytes"]))
            return stream.read().decode("utf-8", errors="replace")
    current = status(directory)
    if action not in {"pause", "cancel"} or not current or current["state"] not in ACTIVE:
        raise ValueError("The attempt is not active")
    if action == "pause" and not read(directory / "spec.json").get("resume_command"):
        raise ValueError("This job has no resume_command")
    path = directory / "request.json"
    if read(path, {}).get("action") != "cancel":
        write(path, dict(action=action))
    return current


def attempt_dir(root, attempt):
    if not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise ValueError("Invalid attempt ID")
    return root / attempt


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(Path(sys.argv[2]))
    else:
        try:
            print("SSHGPU_RPC " + json.dumps({"result": rpc(json.load(sys.stdin))}))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            print("SSHGPU_RPC " + json.dumps({"error": str(exc)}))
