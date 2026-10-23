"""Offline downloader regression tests; never import a real model backend."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "script" / "download.py"


def load_downloader(include_tensorflow=True):
    backend = ModuleType("transformers")
    backend.__version__ = "test"
    backend.AutoConfig = Mock()
    backend.AutoModel = Mock()
    backend.AutoTokenizer = Mock()
    if include_tensorflow:
        backend.TFAutoModel = Mock()
    spec = importlib.util.spec_from_file_location("downloader_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"transformers": backend}):
        spec.loader.exec_module(module)
    return module, backend


class DownloaderTests(unittest.TestCase):
    def test_pytorch_does_not_require_tensorflow_auto_class(self):
        downloader, backend = load_downloader(include_tensorflow=False)
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(downloader.transformers_dowloader("model", Path(directory), is_tf=False))
        backend.AutoModel.from_pretrained.assert_called_once_with("model")

    def test_namespaced_model_creates_nested_output(self):
        downloader, _ = load_downloader()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "models"
            self.assertTrue(downloader.mkdir(output))
            self.assertTrue(downloader.transformers_dowloader("org/model", output, is_tf=False))
            self.assertTrue((output / "org" / "model").is_dir())

    def test_default_requirement_is_checked_in_file(self):
        downloader, _ = load_downloader()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", [str(SCRIPT), "--output-dir", directory]), patch.object(
                downloader, "transformers_dowloader", return_value=True
            ) as download:
                self.assertEqual(downloader.main(), 0)
        self.assertGreater(download.call_count, 0)
        self.assertEqual(download.call_args_list[0].args[0], "bert-base-uncased")

    def test_failed_save_returns_failure_status(self):
        downloader, backend = load_downloader()
        backend.AutoModel.from_pretrained.return_value.save_pretrained.side_effect = OSError("disk full")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", [str(SCRIPT), "-m", "model", "--output-dir", directory]):
                self.assertEqual(downloader.main(), 1)

    def test_tensorflow_is_imported_only_when_requested(self):
        downloader, backend = load_downloader()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"transformers": backend}):
            self.assertTrue(downloader.transformers_dowloader("model", Path(directory), is_tf=True))
        backend.TFAutoModel.from_pretrained.assert_called_once_with("model")
        backend.AutoModel.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
