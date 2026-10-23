"""Offline regression tests for the hfrs batch wrapper."""

import builtins
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import TestCase, main
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "script" / "download.py"


def load_downloader():
    spec = importlib.util.spec_from_file_location("downloader_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def reject_framework_import(name, *args, **kwargs):
        if name.split(".", maxsplit=1)[0] in {"transformers", "torch", "tensorflow"}:
            raise AssertionError(f"unexpected framework import: {name}")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=reject_framework_import):
        spec.loader.exec_module(module)
    return module


class DownloaderTests(TestCase):
    def test_single_model_uses_official_cli_and_namespaced_output(self):
        downloader = load_downloader()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", [str(SCRIPT), "--model", "org/model", "--output-dir", directory]
        ), patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(downloader.main(), 0)

        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["hfrs", "download"])
        self.assertEqual(command[2], "org/model")
        self.assertEqual(command[3], "--local-dir")
        self.assertEqual(command[4], str((Path(directory) / "org" / "model").resolve()))
        self.assertIs(run.call_args.kwargs["check"], False)

    def test_include_patterns_are_passed_through(self):
        downloader = load_downloader()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys,
            "argv",
            [
                str(SCRIPT),
                "--model",
                "gpt2",
                "--include",
                "config.json",
                "--include",
                "tokenizer.json",
                "--output-dir",
                directory,
            ],
        ), patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(downloader.main(), 0)

        self.assertEqual(
            run.call_args.args[0][-4:],
            ["--include", "config.json", "--include", "tokenizer.json"],
        )

    def test_requirement_file_ignores_blank_and_comment_lines(self):
        downloader = load_downloader()
        with tempfile.TemporaryDirectory() as directory:
            requirement = Path(directory) / "models.txt"
            requirement.write_text("# comment\n\norg/first\r\nsecond\n  # comment\n", encoding="utf-8")
            with patch.object(
                sys, "argv", [str(SCRIPT), "--requirement", str(requirement), "--output-dir", directory]
            ), patch.object(
                subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 0),
                    subprocess.CompletedProcess([], 0),
                ],
            ) as run:
                self.assertEqual(downloader.main(), 0)

        self.assertEqual([item.args[0][2] for item in run.call_args_list], ["org/first", "second"])

    def test_failure_continues_and_returns_nonzero(self):
        downloader = load_downloader()
        with tempfile.TemporaryDirectory() as directory:
            requirement = Path(directory) / "models.txt"
            requirement.write_text("first\norg/second\n", encoding="utf-8")
            with patch.object(
                sys, "argv", [str(SCRIPT), "--requirement", str(requirement), "--output-dir", directory]
            ), patch.object(
                subprocess,
                "run",
                side_effect=[
                    subprocess.CompletedProcess([], 17),
                    subprocess.CompletedProcess([], 0),
                ],
            ) as run:
                self.assertEqual(downloader.main(), 1)

        self.assertEqual(run.call_count, 2)

    def test_path_traversal_model_is_rejected(self):
        downloader = load_downloader()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", [str(SCRIPT), "--model", "org/../../outside", "--output-dir", directory]
        ), patch.object(subprocess, "run") as run:
            self.assertEqual(downloader.main(), 1)
        run.assert_not_called()

    def test_missing_hfrs_is_reported_as_failure(self):
        downloader = load_downloader()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "argv", [str(SCRIPT), "--model", "model", "--output-dir", directory]
        ), patch.object(subprocess, "run", side_effect=FileNotFoundError("hfrs")):
            self.assertEqual(downloader.main(), 1)

    def test_tensorflow_mode_is_rejected_with_migration_error(self):
        downloader = load_downloader()
        with patch.object(sys, "argv", [str(SCRIPT), "--model", "model", "--tensorflow"]):
            with self.assertRaises(SystemExit) as error:
                downloader.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    main()
