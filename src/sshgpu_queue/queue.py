import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .ssh import command as ssh_command
from .worker import ACTIVE, lease, read, write

WORKER = Path(__file__).with_name("worker.py").read_text()


class RemoteError(RuntimeError):
    pass


def remote(node, action, **fields):
    command = ssh_command(node, [node["python"], "-c", WORKER])
    payload = dict(root=node["root"], idle_mib=node["idle_mib"], action=action, **fields)
    try:
        process = subprocess.run(command, input=json.dumps(payload), text=True,
                                 capture_output=True, timeout=30, check=True)
        line = next((line for line in reversed(process.stdout.splitlines())
                     if line.startswith("SSHGPU_RPC ")), None)
        if line is None:
            raise RemoteError("No reply from the SSH worker")
        reply = json.loads(line[len("SSHGPU_RPC "):])
        if "error" in reply:
            raise RemoteError(reply["error"])
        return reply["result"]
    except subprocess.CalledProcessError as exc:
        raise RemoteError((exc.stderr or str(exc)).strip()) from exc
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
        raise RemoteError(str(exc)) from exc


class Queue:
    def __init__(self, nodes, state_dir):
        self.nodes = nodes
        self.state_dir = state_dir
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = state_dir / "queue.json"

    @contextmanager
    def edit(self):
        with lease(self.state_dir / "queue.lock", blocking=True):
            jobs = read(self.path, [])
            yield jobs
            write(self.path, jobs)

    def submit(self, spec):
        job = dict(id=uuid4().hex, submitted_at=time.time(), state="queued", spec=spec,
                   attempts=[])
        with self.edit() as jobs:
            jobs.append(job)
        return job

    @staticmethod
    def find(jobs, prefix):
        matches = [job for job in jobs if job["id"].startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one job matching {prefix!r}; found {len(matches)}")
        return matches[0]

    def refresh(self, jobs):
        groups = {}
        for job in jobs:
            if job["state"] in ACTIVE:
                attempt = job["attempts"][-1]
                key = json.dumps(attempt["node"], sort_keys=True)
                groups.setdefault(key, []).append(job)
        for group in groups.values():
            node = group[0]["attempts"][-1]["node"]
            try:
                remote_states = remote(node, "statuses", attempts=[j["attempts"][-1]["id"] for j in group])
            except RemoteError as exc:
                for job in group:
                    job["error"] = str(exc)
                continue
            for job in group:
                attempt = job["attempts"][-1]
                result = remote_states.get(attempt["id"])
                if result is None:
                    job["error"] = "Remote attempt status is missing; placement is still reserved locally."
                else:
                    self.update(job, result)

    @staticmethod
    def update(job, result):
        job["attempts"][-1]["remote"] = result
        job["state"] = result["state"]
        job.pop("error", None)

    def launch(self, job, jobs):
        attempt = job["attempts"][-1]
        try:
            result = remote(attempt["node"], "start", attempt=attempt["id"],
                            gpus=attempt["gpus"], spec=attempt["launch_spec"], worker=WORKER)
        except RemoteError as exc:
            # The SSH reply could have been lost after launch. Keep the intent
            # and retry the same attempt on the same host.
            job["error"] = str(exc)
            write(self.path, jobs)
            return
        if result["state"] == "busy":
            job["attempts"].pop()
            job["state"] = "queued"
        else:
            self.update(job, result)
        write(self.path, jobs)

    def choices(self, job):
        if "resume_node" in job:
            return [job["resume_node"]]
        node = job["spec"].get("node")
        if node is None:
            return list(self.nodes.values())
        return [self.nodes[node]] if node in self.nodes else []

    def tick(self):
        with self.edit() as jobs:
            self.refresh(jobs)
            for job in jobs:
                if job["state"] == "starting":
                    self.launch(job, jobs)
            queued = sorted((job for job in jobs if job["state"] == "queued"),
                            key=lambda job: (-job["spec"]["priority"], job["submitted_at"]))
            available = {}
            for job in queued:
                job.pop("error", None)
                spec = job["spec"]
                choices = self.choices(job)
                if not choices:
                    job["error"] = "The pinned node is no longer configured."
                for node in choices:
                    key = json.dumps(node, sort_keys=True)
                    if key not in available:
                        try:
                            devices = remote(node, "probe")
                            available[key] = {gpu["index"] for gpu in devices if gpu["available"]}
                        except RemoteError as exc:
                            available[key] = set()
                            job["error"] = str(exc)
                    candidates = available[key]
                    if node["gpus"] is not None:
                        candidates = candidates.intersection(node["gpus"])
                    gpus = spec.get("gpus", sorted(candidates)[:spec["gpu_count"]])
                    if len(gpus) != spec["gpu_count"] or not set(gpus).issubset(candidates):
                        continue
                    available[key].difference_update(gpus)
                    launch_spec = dict(spec, job_id=job["id"], argv=spec["command"])
                    if "checkpoint" in job:
                        launch_spec.update(argv=spec["resume_command"], checkpoint=job["checkpoint"])
                    job["attempts"].append(dict(id=uuid4().hex, node=node, gpus=gpus, launch_spec=launch_spec))
                    job["state"] = "starting"
                    write(self.path, jobs)
                    self.launch(job, jobs)
                    if job["state"] != "queued":
                        break

    def run(self, interval, once=False):
        with lease(self.state_dir / "controller.lock") as fd:
            if fd is None:
                raise ValueError("A controller is already running for this state directory")
            while True:
                self.tick()
                if once:
                    return
                time.sleep(interval)

    def change(self, prefix, action, priority=None):
        with self.edit() as jobs:
            job = self.find(jobs, prefix)
            self.refresh([job])
            if action == "promote":
                if job["state"] != "queued":
                    raise ValueError("Only queued jobs can be promoted")
                job["spec"]["priority"] = priority if priority is not None else max(
                    j["spec"]["priority"] for j in jobs if j["state"] == "queued") + 1
            elif action == "resume":
                if job["state"] != "paused":
                    raise ValueError("Only a job with an acknowledged checkpoint can be resumed")
                last = job["attempts"][-1]
                job.update(state="queued", checkpoint=last["remote"]["checkpoint"],
                           resume_node=last["node"], submitted_at=time.time())
            elif action == "cancel" and job["state"] in {"queued", "paused"}:
                job["state"] = "cancelled"
            else:
                if job["state"] not in {"running", "pausing", "cancelling"}:
                    raise ValueError(f"Cannot {action} a job in state {job['state']}")
                attempt = job["attempts"][-1]
                remote(attempt["node"], action, attempt=attempt["id"])
            return job
