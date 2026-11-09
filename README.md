# CardBaton

Hand idle GPUs to the next job in your queue. CardBaton schedules experiments across NVIDIA GPU machines you already reach over SSH. Pin a job to physical GPU indices or let the queue choose cards. A cooperating training job can save a checkpoint, hand off its cards, and resume later.

The quick start below is for operators. [AGENTS.md](AGENTS.md) covers the CLI workflow, queue states and checkpoint contract for AI assistants and contributors.

## Quick start

The controller needs Python 3.9+, OpenSSH and a Unix filesystem. Workers need Linux, Python 3.9+ and `nvidia-smi` on their SSH command path. The controller sends a small standard-library worker over SSH at launch. Put your code, datasets and training environment on the target machine before submitting a job.

If upgrading from the `sshgpu-queue` Python distribution, uninstall it before installing CardBaton. Both `cardbaton` and the `sshgpu` compatibility alias work with existing configs, queue files, and checkpoint scripts.

```sh
python3 -m pip install .
cp examples/cluster.yaml config.yaml
# Edit the SSH destinations and GPU pools in config.yaml.
cardbaton gpus
cardbaton run
```

In a second terminal using the same config:

```sh
cardbaton submit --node lab-a --gpus 1 --priority 10 --timeout 7200 \
  --cwd '~/experiments' -- python3 train.py --seed 0
cardbaton submit --gpu-count 2 -- python3 train.py --seed 1
cardbaton status
cardbaton logs JOB_ID
cardbaton promote JOB_ID
```

Use `cardbaton --config /path/to/config.yaml` when working outside the config directory. `cardbaton submit --job examples/job.yaml` accepts a job definition, and placement, priority and time-limit flags can override its fields. Run `cardbaton run --once` to perform one scheduling pass. Keep `cardbaton run` running to dispatch queued jobs automatically. Launched workers keep running when the controller exits, and restarting it reads their existing attempt state.

The config maps node names to an OpenSSH alias or `user@host`. Put connection settings such as `Port`, `ProxyJump`, and `IdentityFile` in `~/.ssh/config`. The optional node field `gpus` restricts eligible physical indices, and `idle_mib` sets the maximum reported memory in use for an idle GPU (default 64 MiB). A local node can use `local: true` in place of `ssh`. The `remote_state_dir` defaults to `~/.local/state/sshgpu-queue`; use one state directory per remote login across controllers so GPU leases are shared.

### Change the resource pool while the controller runs

Edit the same config file used by `cardbaton run`. The controller reloads it before each scheduling pass. You can add or remove nodes, change their `gpus`, `idle_mib`, SSH target or Python interpreter, and change `poll_seconds`. For example, `gpus: [1, 2, 4, 6, 7]` selects a non-contiguous pool; range strings such as `"1-2,4,6-7"` are not accepted.

Set a node's `gpus: []` to stop new placements there while keeping existing tasks monitored. Restore its indices to enable placements again. Job-level `gpus` must still be nonempty.

Changes affect new placements. Existing starting/running attempts retain their recorded node and GPU allocation and continue to be monitored, even when that node or card is removed from the pool. A queued job pinned to a removed node or excluded GPU remains queued with an explanatory error. A resumed job uses the current GPU pool on its original node; changing/removing that node's SSH target blocks resume rather than moving its checkpoint to another host. New SSH nodes must be reachable; use `bootstrap` separately if first-use key setup is needed.

A malformed or temporarily missing config keeps the last valid settings and reports an error without stopping the controller. Correcting the file applies it on a later pass. `state_dir` cannot be moved by hot reload, and an existing node's `remote_state_dir` cannot change while it has active attempts: those paths hold the queue and live GPU leases. Stop/migrate the relevant state separately when relocating it.

## Set up a public key on first use

```sh
cardbaton run --public-key ~/.ssh/id_ed25519.pub
# Or install a key on one node before running the queue.
cardbaton bootstrap --node lab-a --public-key ~/.ssh/id_ed25519.pub
```

You can also set `public_key: ~/.ssh/id_ed25519.pub` on each SSH node in `config.yaml`. This makes the key choice persist across commands. On startup, the CLI uses SSH authentication to ensure the public key is present in the remote account's `~/.ssh/authorized_keys`. It preserves existing entries and skips matching keys. Run the first setup in a terminal if SSH needs to prompt for a password or trust a host key. The corresponding private key stays on the controller, either next to the `.pub` file or in `ssh-agent`. Load passphrase-protected keys into `ssh-agent` for unattended jobs. Initial login and remote public-key authentication must be allowed by the server.

## Priority and checkpoint recovery

Higher-priority **eligible** jobs run first, with FIFO among equal priorities. `promote` moves a queued job above the current queue, or accepts `--priority N`. To free a card for an urgent job, promote the queued job and request a checkpoint pause for the running job that owns that card.

```sh
cardbaton promote URGENT_JOB
cardbaton pause RUNNING_JOB
cardbaton status
cardbaton resume RUNNING_JOB
```

For a job that declares `resume_command`, the training program must follow this small contract:

1. Poll the path in `SSHGPU_PAUSE_FILE` at a safe point.
2. When the file appears, save the application state and atomically publish `{"checkpoint": "/path/to/saved/state"}` at `SSHGPU_CHECKPOINT_READY`.
3. Exit with status `75` after the save completes. The worker records `paused` once it sees both the exit and a valid checkpoint marker.
4. On resume, load the path in `SSHGPU_CHECKPOINT`. The controller runs `resume_command` on the same node to preserve checkpoint locality.

`examples/checkpoint_job.py` shows a complete cooperative pause and resume. Training code decides what to serialize, such as model and optimizer state, RNG, dataset position or replay buffer. A manual pause leaves the worker in `pausing` until the program acknowledges it. `timeout_seconds` is a wall-clock limit per attempt. At the limit a resumable job gets `checkpoint_grace_seconds` (default 60) to save and exit. `cancel` requests termination of a managed job or removes a queued one from consideration. Logs remain available per attempt via `cardbaton logs JOB_ID --attempt 0`.

## Runtime model

- Each job runs on one node and may use several GPUs on that node. `CUDA_VISIBLE_DEVICES` is set using physical GPU UUIDs. CUDA exposes the selected devices to the process as local ordinals.
- The worker also provides `SSHGPU_GPU_INDICES` (comma-separated physical indices), `SSHGPU_ATTEMPT_ID`, and `SSHGPU_ATTEMPT_DIR`. Use physical indices for tools that require them, such as CPU-affinity or EGL configuration; use CUDA's local ordinals for tensor device selection. The attempt directory contains `job.log` and the worker's status files. These worker-assigned values override any same-named entries in the job's `env`.
- Allocation checks GPU memory and compute processes, then acquires host-local leases and checks again before launch. Leases coordinate workers using the same state directory. Independent processes that start GPU work outside the queue do not participate in its leases.
- Queue and attempt state persist on disk. If the SSH launch reply is lost, the controller retries the same remote attempt ID instead of creating a new placement. GPU leases stay open for the worker and foreground training process. A worker that disappears without a final status is reported as `lost`.
- This release uses a single controller per queue and trusted SSH accounts. Running commands and job definitions have the same privileges as that remote account. A checkpoint pause requires application cooperation. It does not capture GPU memory or transfer checkpoints between nodes.

## Related projects

- [gflow](https://github.com/AndPuQing/gflow) provides GPU-aware scheduling for a single node.
- [multi_task](https://github.com/cs-fisha/multi_task) dispatches experiments across SSH hosts and records their artifacts.
- [SkyPilot](https://docs.skypilot.ai/en/latest/reservations/existing-machines.html) can manage jobs on existing machines within its broader provisioning and orchestration stack.

CardBaton focuses on the priority queue and checkpoint handshake for commands running in existing SSH environments.
