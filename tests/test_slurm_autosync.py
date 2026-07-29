import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "slurm_autosync.py"
spec = importlib.util.spec_from_file_location("slurm_autosync", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.cluster_data = json.loads((ROOT / "config" / "cluster.example.json").read_text(encoding="utf-8"))
        self.experiment_data = json.loads((ROOT / "config" / "experiment.single.example.json").read_text(encoding="utf-8"))

    def test_example_config_is_valid(self):
        cluster = module.parse_cluster(self.cluster_data)
        module.validate_experiment(self.experiment_data, cluster)

    def test_rejects_unsafe_experiment_name(self):
        self.experiment_data["name"] = "bad; name"
        cluster = module.parse_cluster(self.cluster_data)
        with self.assertRaises(module.ConfigurationError):
            module.validate_experiment(self.experiment_data, cluster)

    def test_rejects_compute_node_download(self):
        self.experiment_data["train_command"] = ["wget", "https://example.invalid/model.pt"]
        cluster = module.parse_cluster(self.cluster_data)
        with self.assertRaises(module.ConfigurationError):
            module.validate_experiment(self.experiment_data, cluster)

    def test_rejects_remote_parent_traversal(self):
        self.cluster_data["remote"]["workspace"] = "/home/user/../other"
        with self.assertRaises(module.ConfigurationError):
            module.parse_cluster(self.cluster_data)


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.cluster = module.parse_cluster(json.loads((ROOT / "config" / "cluster.example.json").read_text(encoding="utf-8")))

    def test_single_render_has_file_checks_and_no_unresolved_values(self):
        experiment = module.validate_experiment(
            json.loads((ROOT / "config" / "experiment.single.example.json").read_text(encoding="utf-8")), self.cluster
        )
        rendered = module.render_job(ROOT, self.cluster, experiment)
        self.assertIn("set -euo pipefail", rendered)
        self.assertIn("Required file missing", rendered)
        self.assertNotIn("{{", rendered)
        self.assertNotIn("wget", rendered)

    def test_parallel_trial_count_is_dynamic_and_gpu_is_slurm_managed(self):
        experiment = module.validate_experiment(
            json.loads((ROOT / "config" / "experiment.parallel.example.json").read_text(encoding="utf-8")), self.cluster
        )
        rendered = module.render_job(ROOT, self.cluster, experiment)
        self.assertIn("#SBATCH --ntasks=2", rendered)
        self.assertIn("--gpus-per-task=1", rendered)
        self.assertNotIn("DEVICE=$SLURM_PROCID", rendered)
        self.assertNotIn("device=$SLURM_PROCID", rendered)


class DatasetAndResultsTests(unittest.TestCase):
    def test_yolo_dataset_pairing(self):
        profile = json.loads((ROOT / "config" / "dataset.yolo.example.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datasets"
            for relative in ("train/images", "train/labels", "val/images", "val/labels"):
                (root / relative).mkdir(parents=True)
            for split in ("train", "val"):
                (root / split / "images" / "one.jpg").write_bytes(b"")
                (root / split / "labels" / "one.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            self.assertEqual(module.local_dataset_check(Path(directory), profile), [])
            (root / "train" / "labels" / "one.txt").unlink()
            self.assertTrue(module.local_dataset_check(Path(directory), profile))

    def test_results_summary_recognizes_yolo_columns(self):
        fixture = ROOT / "tests" / "fixtures" / "results.csv"
        summary = module.read_csv_summary(fixture)
        self.assertEqual(summary["epochs"], 3)
        self.assertAlmostEqual(summary["map50"]["best"], 0.72)
        self.assertEqual(summary["map50"]["best_epoch"], 2)
        self.assertAlmostEqual(summary["precision"]["final"], 0.70)


if __name__ == "__main__":
    unittest.main()
