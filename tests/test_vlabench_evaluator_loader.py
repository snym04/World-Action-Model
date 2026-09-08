"""CPU import regression; no simulator assets or other VLA model stack."""
import runpy
import tempfile
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "inference/vlabench_policy/evaluate_flowwam.py"

class EvaluatorLoaderTest(unittest.TestCase):
    def test_unchanged_module_without_parent_initializers(self):
        loader = runpy.run_path(str(ENTRY))["_load_official_evaluator"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "VLABench/evaluation/evaluator"
            folder.mkdir(parents=True)
            for directory in [folder, folder.parent, folder.parent.parent]:
                (directory / "__init__.py").write_text("raise RuntimeError('optional model imported')")
            source = folder / "base.py"
            source.write_text("class Evaluator:\n    sentinel = 42\n")
            before = source.read_bytes()
            self.assertEqual(loader(root).sentinel, 42)
            self.assertEqual(source.read_bytes(), before)

    def test_missing_source_is_an_error(self):
        loader = runpy.run_path(str(ENTRY))["_load_official_evaluator"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                loader(Path(tmp))

if __name__ == "__main__":
    unittest.main()
