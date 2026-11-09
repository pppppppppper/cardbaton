import json
import sys
import tempfile
import unittest
from pathlib import Path

from cardbaton.worker import read, run, write


class WorkerEnvironmentTests(unittest.TestCase):
    def test_worker_exports_authoritative_placement_and_attempt(self):
        for indices, uuids in [([2], ["GPU-test-a"]),
                               ([2, 5], ["GPU-test-a", "GPU-test-b"])]:
            with self.subTest(indices=indices), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp) / ("a" * 32)
                directory.mkdir()
                output = directory / "observed.json"
                names = ["CUDA_VISIBLE_DEVICES", "SSHGPU_GPU_INDICES",
                         "SSHGPU_ATTEMPT_ID", "SSHGPU_ATTEMPT_DIR", "SSHGPU_JOB_ID"]
                program = (
                    "import json,os,pathlib,sys; "
                    "pathlib.Path(sys.argv[1]).write_text("
                    "json.dumps({key:os.environ[key] for key in sys.argv[2:]}))"
                )
                write(directory / "spec.json", {
                    "argv": [sys.executable, "-c", program, str(output), *names],
                    "cwd": temp,
                    "job_id": "test-job",
                    "gpu_uuids": uuids,
                    "gpu_fds": [],
                    "env": {name: "must-not-override-worker" for name in names},
                })
                write(directory / "status.json", {"state": "starting", "gpus": indices})

                # A real CPU subprocess receives simulated placement metadata;
                # no NVIDIA devices, SSH server or application files are used.
                run(directory)

                self.assertEqual(read(directory / "status.json")["state"], "succeeded")
                self.assertEqual(json.loads(output.read_text()), {
                    "CUDA_VISIBLE_DEVICES": ",".join(uuids),
                    "SSHGPU_GPU_INDICES": ",".join(map(str, indices)),
                    "SSHGPU_ATTEMPT_ID": directory.name,
                    "SSHGPU_ATTEMPT_DIR": str(directory.resolve()),
                    "SSHGPU_JOB_ID": "test-job",
                })


if __name__ == "__main__":
    unittest.main()
