"""Download Hugging Face repository snapshots with the official hfrs CLI."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


REPOSITORY_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


def parse_models(requirement: Path) -> list[str]:
    """Read model IDs, ignoring empty lines and whole-line comments."""
    with requirement.open(encoding="utf-8") as source:
        return [
            line.strip()
            for line in source
            if line.strip() and not line.strip().startswith("#")
        ]


def destination_for(output_root: Path, repo_id: str) -> Path:
    """Return a safe output directory for a public Hub repo ID."""
    components = repo_id.split("/")
    if len(components) > 2 or not all(REPOSITORY_COMPONENT.fullmatch(part) for part in components):
        raise ValueError(f"invalid Hugging Face repository ID: {repo_id!r}")

    destination = (output_root / Path(*components)).resolve()
    try:
        destination.relative_to(output_root)
    except ValueError as error:
        raise ValueError(f"repository output escapes output directory: {repo_id!r}") from error
    return destination


def download_model(output_root: Path, repo_id: str, include: list[str]) -> bool:
    """Delegate one full repository snapshot download to hfrs."""
    try:
        destination = destination_for(output_root, repo_id)
    except ValueError as error:
        print(f"Failed {repo_id}: {error}", file=sys.stderr, flush=True)
        return False

    command = ["hfrs", "download", repo_id, "--local-dir", str(destination)]
    for pattern in include:
        command.extend(("--include", pattern))

    try:
        result = subprocess.run(command, check=False)
    except FileNotFoundError:
        print("Failed to run hfrs: executable not found on PATH", file=sys.stderr, flush=True)
        return False
    except OSError as error:
        print(f"Failed to run hfrs for {repo_id}: {error}", file=sys.stderr, flush=True)
        return False

    if result.returncode != 0:
        print(f"Failed {repo_id}: hfrs exited with status {result.returncode}", file=sys.stderr, flush=True)
        return False

    print(f"Downloaded {repo_id} to {destination}", flush=True)
    return True


def build_parser(project_dir: Path) -> argparse.ArgumentParser:
    default_output_dir = os.environ.get("HF_DOWNLOADER_OUTPUT_DIR", str(project_dir / "models"))
    parser = argparse.ArgumentParser(
        description="Batch-download Hugging Face repositories with the official hfrs CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="folder in which repository snapshots are saved",
    )
    parser.add_argument(
        "--requirement",
        "-r",
        type=Path,
        default=project_dir / "configs" / "download.txt",
        help="file containing one repository ID per line",
    )
    parser.add_argument("--model", "-m", help="download one repository instead of the list")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="only download files matching this glob (repeatable)",
    )
    # Retain a focused migration error for scripts that used the retired flag.
    parser.add_argument("--tensorflow", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    parser = build_parser(project_dir)
    args = parser.parse_args()
    if args.tensorflow:
        parser.error(
            "--tensorflow is retired: hfrs downloads repository files and does not convert or save TensorFlow models"
        )

    try:
        model_ids = [args.model] if args.model is not None else parse_models(args.requirement)
    except OSError as error:
        print(f"Failed to read repository list {args.requirement}: {error}", file=sys.stderr)
        return 1

    output_root = args.output_dir.resolve()
    print(f"Plan: download {len(model_ids)} repositories to {output_root}", flush=True)
    successful = True
    for repo_id in model_ids:
        if not download_model(output_root, repo_id, args.include):
            successful = False
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
