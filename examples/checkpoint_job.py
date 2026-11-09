"""Copy this file to the job's working directory on the worker."""

import argparse
import json
import os
import time
from pathlib import Path


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--delay", type=float, default=1)
    args = parser.parse_args()
    pause = Path(os.environ["SSHGPU_PAUSE_FILE"])
    marker = Path(os.environ["SSHGPU_CHECKPOINT_READY"])
    restore = os.environ.get("SSHGPU_CHECKPOINT")
    step = json.loads(Path(restore).read_text())["step"] if restore else 0
    checkpoint = Path(f"checkpoint-{os.environ['SSHGPU_JOB_ID']}.json").resolve()
    print(f"Starting at step {step}", flush=True)
    while step < args.steps:
        time.sleep(args.delay)
        step += 1
        if pause.exists():
            save_json(checkpoint, {"step": step})
            save_json(marker, {"checkpoint": str(checkpoint)})
            print(f"Checkpoint saved at step {step}", flush=True)
            return 75
    print(f"Finished at step {step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
