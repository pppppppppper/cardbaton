import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from cardbaton.config import load_config
from cardbaton.queue import Queue


class ConfigReloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.yaml"
        self.raw = {
            "state_dir": "state", "poll_seconds": 5,
            "nodes": {"a": {"local": True, "gpus": [0, 1],
                            "remote_state_dir": str(self.root / "worker")}},
        }
        self.save()
        nodes, directory, interval = load_config(self.config)
        self.queue = Queue(nodes, directory)
        self.interval = interval

    def save(self):
        self.config.write_text(json.dumps(self.raw))

    def reload(self):
        with redirect_stderr(io.StringIO()):
            self.interval = self.queue.reload_config(self.config, self.interval)

    def active_job(self):
        job = self.queue.submit({"command": ["python3", "train.py"], "priority": 0,
                                 "gpu_count": 1, "node": "a"})
        with self.queue.edit() as jobs:
            jobs[0].update(state="running", attempts=[{
                "id": "b" * 32, "node": copy.deepcopy(self.queue.nodes["a"]), "gpus": [0],
            }])
        return job["id"]

    def test_changes_nodes_pool_and_interval(self):
        self.raw["nodes"]["a"]["gpus"] = [1, 3, 5]
        self.raw["nodes"]["b"] = {"local": True, "gpus": [2]}
        self.raw["poll_seconds"] = 2
        self.save()
        self.reload()
        self.assertEqual(self.queue.nodes["a"]["gpus"], [1, 3, 5])
        self.assertIn("b", self.queue.nodes)
        self.assertEqual(self.interval, 2)

    def test_invalid_edit_retains_config_and_recovers(self):
        original = copy.deepcopy(self.queue.nodes)
        self.config.write_text("nodes: [unterminated")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(self.queue.reload_config(self.config, 5), 5)
            self.assertEqual(self.queue.reload_config(self.config, 5), 5)
        self.assertEqual(self.queue.nodes, original)
        self.assertEqual(stderr.getvalue().count("keeping last valid config"), 1)
        self.raw["poll_seconds"] = 3
        self.save()
        self.reload()
        self.assertEqual(self.interval, 3)
        self.assertIsNone(self.queue._config_error)

    def test_removed_node_does_not_retarget_running_attempt(self):
        self.active_job()
        old_node = copy.deepcopy(self.queue.nodes["a"])
        self.raw["nodes"] = {"b": {"local": True, "gpus": [2]}}
        self.save()
        self.reload()
        with self.queue.edit() as jobs:
            self.assertEqual(jobs[0]["attempts"][0]["node"], old_node)
            with patch("cardbaton.queue.remote", return_value={"b" * 32: {"state": "running"}}) as rpc:
                self.queue.refresh(jobs)
                self.assertEqual(rpc.call_args.args[0], old_node)
        self.assertNotIn("a", self.queue.nodes)

    def test_queue_directory_cannot_move(self):
        original = copy.deepcopy(self.queue.nodes)
        self.raw["state_dir"] = "different-state"
        self.raw["poll_seconds"] = 2
        self.save()
        self.reload()
        self.assertEqual(self.queue.nodes, original)
        self.assertEqual(self.interval, 5)

    def test_lease_root_waits_for_active_attempts(self):
        self.active_job()
        root = self.queue.nodes["a"]["root"]
        self.raw["nodes"]["a"]["remote_state_dir"] = str(self.root / "new-worker")
        self.save()
        self.reload()
        self.assertEqual(self.queue.nodes["a"]["root"], root)
        with self.queue.edit() as jobs:
            jobs[0]["state"] = "succeeded"
        self.reload()
        self.assertNotEqual(self.queue.nodes["a"]["root"], root)

    def test_resume_respects_current_pool_and_checkpoint_host(self):
        original = copy.deepcopy(self.queue.nodes["a"])
        job = {"resume_node": original, "spec": {}}
        self.raw["nodes"]["a"]["gpus"] = [5]
        self.save()
        self.reload()
        self.assertEqual(self.queue.choices(job)[0]["gpus"], [5])
        self.queue.nodes["a"]["ssh"] = "different-host"
        self.assertEqual(self.queue.choices(job), [])

    def test_next_placement_uses_updated_pool(self):
        self.queue.submit({"command": ["python3", "train.py"], "node": "a",
                           "gpus": [0], "gpu_count": 1, "priority": 1})
        self.queue.submit({"command": ["python3", "train.py"], "node": "a",
                           "gpu_count": 1, "priority": 0})
        self.raw["nodes"]["a"]["gpus"] = [1]
        self.save()
        self.reload()
        placements = []

        def rpc(node, action, **fields):
            if action == "probe":
                return [{"index": index, "available": True} for index in [0, 1]]
            if action == "start":
                placements.append(fields["gpus"])
                return {"state": "running"}
            raise AssertionError(action)

        with patch("cardbaton.queue.remote", side_effect=rpc):
            self.queue.tick()
        self.assertEqual(placements, [[1]])
        with self.queue.edit() as jobs:
            self.assertEqual(jobs[0]["state"], "queued")
            self.assertIn("outside the current pool", jobs[0]["error"])
            self.assertEqual(jobs[1]["state"], "running")


if __name__ == "__main__":
    unittest.main()
