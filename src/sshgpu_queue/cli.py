import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from .config import load_config, load_yaml, validate_job
from .queue import Queue, RemoteError, remote
from .ssh import bootstrap


def main():
    parser = argparse.ArgumentParser(prog="sshgpu", description="Queue GPU jobs on existing SSH machines")
    parser.add_argument("--config", default="config.yaml")
    commands = parser.add_subparsers(dest="action", required=True)
    submit = commands.add_parser("submit", help="Submit a job YAML or an argv command")
    submit.add_argument("--job")
    submit.add_argument("--node")
    submit.add_argument("--cwd")
    placement = submit.add_mutually_exclusive_group()
    placement.add_argument("--gpus", type=lambda value: [int(item) for item in value.split(",")])
    placement.add_argument("--gpu-count", type=int)
    submit.add_argument("--priority", type=int)
    submit.add_argument("--timeout", type=int, dest="timeout_seconds")
    submit.add_argument("command", nargs=argparse.REMAINDER)
    run = commands.add_parser("run", help="Run the queue controller")
    run.add_argument("--once", action="store_true")
    run.add_argument("--public-key", help="Install this public key on SSH nodes if needed")
    setup = commands.add_parser("bootstrap", help="Install and verify SSH public keys")
    setup.add_argument("--public-key")
    setup.add_argument("--node")
    commands.add_parser("gpus", help="Query GPU availability")
    commands.add_parser("status", help="List job states").add_argument("--json", action="store_true")
    for action in ("promote", "pause", "resume", "cancel", "logs"):
        sub = commands.add_parser(action)
        sub.add_argument("job_id")
        if action == "promote":
            sub.add_argument("--priority", type=int)
        if action == "logs":
            sub.add_argument("--attempt", type=int, default=-1, help="Zero-based attempt index; default is latest")
    args = parser.parse_args()
    try:
        nodes, directory, interval = load_config(args.config)
        queue = Queue(nodes, directory)
        if args.action in {"run", "bootstrap"}:
            target = getattr(args, "node", None)
            if target is not None and target not in nodes:
                raise ValueError(f"Unknown node {target!r}")
            selected = [node for node in nodes.values() if node["ssh"] and
                        (target is None or node["name"] == target)]
            if args.public_key:
                for node in selected:
                    node["public_key"] = str(Path(args.public_key).expanduser().resolve())
            configured = [node for node in selected if node.get("public_key")]
            if args.action == "bootstrap" and not configured:
                raise ValueError("Pass --public-key or configure a public_key on the selected SSH nodes")
            for node in configured:
                print(f"{node['name']}  {bootstrap(node)}", file=sys.stderr)
            if args.action == "bootstrap":
                return 0
        if args.action == "submit":
            spec = load_yaml(args.job) if args.job else {}
            for key in ("node", "cwd", "gpus", "gpu_count", "priority", "timeout_seconds"):
                value = getattr(args, key)
                if value is not None:
                    spec[key] = value
            if args.gpus is not None:
                spec.pop("gpu_count", None)
            if args.gpu_count is not None:
                spec.pop("gpus", None)
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if command:
                spec["command"] = command
            print(queue.submit(validate_job(spec, nodes))["id"])
        elif args.action == "run":
            queue.run(interval, args.once)
        elif args.action == "gpus":
            for name, node in nodes.items():
                try:
                    devices = remote(node, "probe")
                    print(json.dumps({"node": name, "gpus": [gpu for gpu in devices if
                          node["gpus"] is None or gpu["index"] in node["gpus"]]}))
                except RemoteError as exc:
                    print(json.dumps({"node": name, "error": str(exc)}))
        elif args.action == "status":
            with queue.edit() as jobs:
                queue.refresh(jobs)
                if args.json:
                    print(json.dumps(jobs, indent=2))
                else:
                    for job in jobs:
                        attempt = job["attempts"][-1] if job["attempts"] else None
                        placement = f"{attempt['node']['name']}/{attempt['gpus']}" if attempt else "-"
                        print(job["id"][:12], job["state"], f"priority={job['spec']['priority']}",
                              placement, job["spec"].get("name", ""), job.get("error", ""))
        elif args.action == "logs":
            with queue.edit() as jobs:
                job = queue.find(jobs, args.job_id)
                try:
                    attempt = job["attempts"][args.attempt]
                except IndexError as exc:
                    raise ValueError("No matching attempt") from exc
            print(remote(attempt["node"], "logs", attempt=attempt["id"], max_bytes=65536), end="")
        else:
            job = queue.change(args.job_id, args.action, getattr(args, "priority", None))
            print(job["id"], args.action, job["state"])
    except (ValueError, OSError, RemoteError, yaml.YAMLError, subprocess.SubprocessError) as exc:
        print(f"sshgpu: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
