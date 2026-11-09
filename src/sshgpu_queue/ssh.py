"""OpenSSH connection and opt-in first-use public-key installation."""

import shlex
import subprocess
import sys
from pathlib import Path

INSTALL_KEY = """
import fcntl, os, shlex, sys
from pathlib import Path

directory = Path.home() / '.ssh'
directory.mkdir(mode=0o700, exist_ok=True)
key = sys.argv[1]
identity = tuple(key.split()[:2])
fd = os.open(directory / 'authorized_keys', os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
with os.fdopen(fd, 'a+', encoding='utf-8', errors='surrogateescape') as file:
    fcntl.flock(file, fcntl.LOCK_EX)
    file.seek(0)
    contents = file.read()
    for line in contents.splitlines():
        if line.lstrip().startswith('#'):
            continue
        try:
            fields = shlex.split(line)
        except ValueError:
            continue
        if any(tuple(fields[i:i + 2]) == identity for i in range(len(fields) - 1)):
            break
    else:
        prefix = '\\n' if contents and not contents.endswith('\\n') else ''
        file.write(prefix + key + '\\n')
        file.flush()
        os.fsync(file.fileno())
"""


def command(node, arguments, *, bootstrap=False):
    if node["ssh"] is None:
        return arguments
    interactive = bootstrap and sys.stdin.isatty()
    batch = "no" if interactive else "yes"
    prefix = ["ssh", "-T", "-o", f"BatchMode={batch}", "-o", "ConnectTimeout=5",
              "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
    if node.get("public_key") and not bootstrap:
        public = Path(node["public_key"])
        private = public.with_suffix("") if public.suffix == ".pub" else public
        identity = private if private != public and private.is_file() else public
        prefix += ["-o", "IdentitiesOnly=yes", "-i", str(identity)]
    return prefix + [node["ssh"], shlex.join(arguments)]


def bootstrap(node):
    public = Path(node["public_key"])
    key = public.read_text().strip()
    if not key.startswith(("ssh-", "ecdsa-", "sk-")) or len(key.splitlines()) != 1 or len(key.split()) < 2:
        raise ValueError(f"Provide one OpenSSH public key in {public}")
    check = subprocess.run(["ssh-keygen", "-l", "-f", str(public)], capture_output=True, text=True)
    if check.returncode:
        raise ValueError(f"Invalid OpenSSH public key: {public}")
    arguments = [node["python"], "-c", INSTALL_KEY, key]
    install = subprocess.run(command(node, arguments), capture_output=True, timeout=20)
    if install.returncode:
        # The server still authenticates the first connection. OpenSSH handles
        # host verification and any interactive password or passphrase prompt.
        fallback = command(node, arguments, bootstrap=True)
        if subprocess.run(fallback).returncode:
            raise ValueError(f"Could not install the key on {node['name']}; check initial SSH access")
    if subprocess.run(command(node, ["true"]), capture_output=True, timeout=15).returncode:
        raise ValueError(f"Key installed on {node['name']}, but key login failed. Load its private key into ssh-agent or keep it next to the .pub file.")
    return "public key present; noninteractive SSH verified"
