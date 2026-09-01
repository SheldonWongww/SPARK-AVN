import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AvnBlankValSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = ROOT / "avn/scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        cls.entry = _load(
            "avn_blank_val_search",
            scripts / "run_avn_blank_val_search.py",
        )

    @staticmethod
    def _args(model):
        return SimpleNamespace(
            batch_id="unit-{}".format(model),
            gpus=("0", "1", "2", "3"),
            smoke=False,
            smoke_setting="single_source",
            eam_concurrency=4,
            feedtta_concurrency=4,
            atena_concurrency=3,
            idea_concurrency=2,
        )

    def test_each_model_has_six_cells_and_384_jobs(self):
        for model in self.entry.MODELS:
            with self.subTest(model=model):
                self.entry.configure(model)
                runner = self.entry.runner
                spec = runner.load_spec(runner.DEFAULT_SPEC)
                jobs = runner.build_jobs(spec, self._args(model))
                self.assertEqual(len(jobs), 384)
                for method in self.entry.METHODS:
                    self.assertEqual(len(runner.method_points(spec, method)), 64)
                    for setting in runner.SOURCE_SETTINGS:
                        self.assertEqual(
                            sum(
                                job.method == method
                                and job.source_setting == setting
                                for job in jobs
                            ),
                            64,
                        )

    def test_all_adaptation_learning_rates_are_below_one_e_minus_six(self):
        for model in self.entry.MODELS:
            with self.subTest(model=model):
                self.entry.configure(model)
                runner = self.entry.runner
                spec = runner.load_spec(runner.DEFAULT_SPEC)
                for method in self.entry.METHODS:
                    for point in runner.method_points(spec, method):
                        values = (
                            (point["lr_query"], point["lr_self"])
                            if method == "atena"
                            else (point["lr"],)
                        )
                        self.assertTrue(
                            all(float(value) < 1e-6 for value in values),
                            (model, method, point),
                        )

    def test_model_specific_runner_is_selected(self):
        for model, runner_name in (
            ("smt_audio", "eval_smt_audio.sh"),
            ("enmus", "eval_enmus.sh"),
        ):
            with self.subTest(model=model):
                self.entry.configure(model)
                runner = self.entry.runner
                spec = runner.load_spec(runner.DEFAULT_SPEC)
                jobs = runner.build_jobs(spec, self._args(model))
                command = runner.job_command(jobs[0], 2000)
                self.assertEqual(Path(command[1]).name, runner_name)
                self.assertIn(model, str(runner.CHECKPOINTS["single_source"]))


if __name__ == "__main__":
    unittest.main()
