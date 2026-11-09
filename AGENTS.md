# Working with SSH GPU Queue

Read [README.md](README.md) for installation, node configuration, and the training-side checkpoint example. The CLI is `sshgpu`, the controller runs on a Unix machine, and each job runs on one Linux worker with access to one or more NVIDIA GPUs.

## Operate a queue

Use the user's existing config and the same `--config` path for every command. [examples/cluster.yaml](examples/cluster.yaml) contains example hostnames. Use real SSH destinations and GPU pools from the user's config.

| Intent | Command |
| --- | --- |
| Inspect GPU availability | `sshgpu --config config.yaml gpus` |
| Inspect queue and attempt state | `sshgpu --config config.yaml status --json` |
| Read logs from the latest attempt | `sshgpu --config config.yaml logs JOB_ID` |
| Submit a job definition | `sshgpu --config config.yaml submit --job job.yaml` |
| Move a queued job ahead | `sshgpu --config config.yaml promote JOB_ID` |
| Request a checkpoint pause | `sshgpu --config config.yaml pause JOB_ID` |
| Requeue a paused job | `sshgpu --config config.yaml resume JOB_ID` |

`submit` prints a job ID and queues the job. A running `sshgpu run` controller dispatches it when eligible GPUs are available. `run --once` dispatches eligible jobs for one pass. It can pick up other jobs already in the same queue. A worker keeps running when the controller exits.

Use `status --json` to distinguish queue state from remote attempt state. `queued` is waiting for placement. `starting` means a placement attempt has been recorded. `pausing` means the worker has requested a checkpoint. `paused` confirms the job exited with a checkpoint marker. `lost` means the worker exited without reporting a final status. Report a placement or checkpoint as confirmed only after the corresponding state appears.

The CLI can submit commands directly with `--node`, `--gpus` or `--gpu-count`, `--priority`, `--timeout` (seconds), and `--cwd`. Use `--` before the remote command. [examples/job.yaml](examples/job.yaml) shows how to declare `resume_command` for a resumable job.

## SSH access and interruption

`bootstrap` and `run --public-key` install a public key in the configured SSH account's `authorized_keys`. Per-node `public_key` in the config enables the same setup on controller startup. Initial authentication may require an interactive terminal. Keep private keys in the local SSH setup.

`promote` changes the priority of a queued job. For a running job, `pause` is a separate operation. The training program must observe `SSHGPU_PAUSE_FILE`, save its state, atomically write a checkpoint path to `SSHGPU_CHECKPOINT_READY`, and exit with code `75`. Once the worker reports `paused`, `resume` runs the configured resume command on the original node with `SSHGPU_CHECKPOINT` set. See [examples/checkpoint_job.py](examples/checkpoint_job.py).

## Change the code

| Area | File |
| --- | --- |
| CLI commands and output | [cli.py](src/sshgpu_queue/cli.py) |
| YAML and job validation | [config.py](src/sshgpu_queue/config.py) |
| Controller, queue state, and SSH RPC | [queue.py](src/sshgpu_queue/queue.py) |
| Worker, GPU leases, and checkpoint handling | [worker.py](src/sshgpu_queue/worker.py) |
| OpenSSH and first-use key setup | [ssh.py](src/sshgpu_queue/ssh.py) |

The controller sends `worker.py` to workers and expects it to run with the Python standard library. Preserve attempt IDs across uncertain SSH launch replies, and preserve GPU leases until the worker and foreground training process release them. For scheduler tests, use an isolated state directory and a simulated `nvidia-smi` inventory. Keep site-specific hostnames, job logs, checkpoints, and SSH credentials out of examples and commits.
