from pathlib import Path

import yaml


def mapping(value, field):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a mapping with string keys")
    return value


def keys(value, allowed, field):
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {field} fields: {', '.join(sorted(unknown))}")


def integer(value, field, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid integer for {field}: {value!r}")
    return value


def string(value, field):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def indices(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list of GPU indices")
    result = [integer(index, field, 0) for index in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains a duplicate GPU index")
    return result


def load_yaml(path):
    return mapping(yaml.safe_load(Path(path).expanduser().read_text()), str(path))


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = load_yaml(path)
    keys(config, {"state_dir", "poll_seconds", "nodes"}, "config")
    nodes = mapping(config.get("nodes"), "nodes")
    if not nodes:
        raise ValueError("Configure at least one node")
    normalized = {}
    for name, raw in nodes.items():
        node = mapping(raw, f"nodes.{name}")
        keys(node, {"ssh", "local", "python", "remote_state_dir", "gpus", "idle_mib", "public_key"}, "node")
        local = node.get("local", False)
        if type(local) is not bool or local == ("ssh" in node):
            raise ValueError(f"Node {name!r} needs either ssh or local: true")
        target = None if local else string(node["ssh"], "ssh")
        if target and (target.startswith("-") or any(c.isspace() for c in target)):
            raise ValueError("ssh must be an OpenSSH alias or user@host; configure options in ~/.ssh/config")
        interpreter = string(node.get("python", "python3"), "python")
        if interpreter.startswith("-"):
            raise ValueError("python must be an executable name or path")
        normalized[name] = dict(
            name=name, ssh=target, python=interpreter,
            root=string(node.get("remote_state_dir", "~/.local/state/sshgpu-queue"), "remote_state_dir"),
            gpus=indices(node["gpus"], "node.gpus") if "gpus" in node else None,
            idle_mib=integer(node.get("idle_mib", 64), "idle_mib", 0),
        )
        if "public_key" in node:
            if local:
                raise ValueError(f"Node {name!r} is local and does not use SSH keys")
            public_key = Path(string(node["public_key"], "public_key")).expanduser()
            normalized[name]["public_key"] = str((path.parent / public_key).resolve())
    state_dir = path.parent / Path(string(config.get("state_dir", ".sshgpu"), "state_dir")).expanduser()
    return normalized, state_dir.resolve(), integer(config.get("poll_seconds", 5), "poll_seconds", 1)


def argv(value, field):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty argument list")
    string(value[0], field)
    if any(not isinstance(arg, str) or "\x00" in arg for arg in value):
        raise ValueError(f"{field} must contain only strings")
    return value


def validate_job(raw, nodes):
    keys(raw, {"name", "command", "cwd", "env", "node", "gpus", "gpu_count", "priority",
               "timeout_seconds", "resume_command", "checkpoint_grace_seconds"}, "job")
    spec = raw.copy()
    spec["command"] = argv(spec.get("command"), "command")
    if "resume_command" in spec:
        argv(spec["resume_command"], "resume_command")
    for field in ("name", "cwd"):
        if field in spec:
            string(spec[field], field)
    if "node" in spec:
        string(spec["node"], "node")
        if spec["node"] not in nodes:
            raise ValueError(f"Unknown node {spec['node']!r}")
    if "gpus" in spec:
        indices(spec["gpus"], "job.gpus")
        if "gpu_count" in spec:
            raise ValueError("Specify gpus or gpu_count, not both")
        spec["gpu_count"] = len(spec["gpus"])
    else:
        spec["gpu_count"] = integer(spec.get("gpu_count", 1), "gpu_count", 1)
    spec["priority"] = integer(spec.get("priority", 0), "priority")
    for field in ("timeout_seconds", "checkpoint_grace_seconds"):
        if field in spec:
            integer(spec[field], field, 1)
    for key, value in mapping(spec.get("env", {}), "env").items():
        if not key or "=" in key or "\x00" in key or not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"Invalid environment entry {key!r}; quote YAML values")
    eligible = [node for name, node in nodes.items() if name == spec.get("node", name)]
    if not any(node["gpus"] is None or (len(node["gpus"]) >= spec["gpu_count"] and
               set(spec.get("gpus", ())).issubset(node["gpus"])) for node in eligible):
        raise ValueError("The requested GPUs are outside the eligible nodes' configured pools")
    return spec
