# download-transformers-pretrained

Batch-download model repository snapshots from Hugging Face with the official
Rust CLI, `hfrs`. The Python entry point uses only the standard library; it
delegates network transfer, authentication, caching, retries, and file selection
to `hfrs`.

## Install the official CLI

This project uses the verified `hfrs` 1.0.0 source revision. Install it with a
Rust toolchain and Cargo:

```sh
cargo install --locked \
  --git https://github.com/huggingface/hf-hub.git \
  --rev 473de9430b0fbc544fb1d91102180eb713f8dc87 \
  hfrs
```

Ensure Cargo's `bin` directory is on `PATH`. `hfrs` uses the official Hub CLI
authentication settings, including `HF_TOKEN` for gated or private repositories.

## Usage

Edit `configs/download.txt` to choose repositories, then run:

```sh
python script/download.py
```

Download one repository or filter downloaded files:

```sh
python script/download.py --model org/model --output-dir ./models
python script/download.py --model gpt2 --include config.json --output-dir ./models
```

The list file accepts one repository ID per line; blank lines and lines starting
with `#` are ignored. `--include` may be repeated and applies to every repository
in the selected list. The default output directory is `./models` beside this
repository, and each snapshot is saved under `<output-dir>/<repo-id>` (for
example, `models/org/model`). The wrapper attempts every listed repository and
exits nonzero if any download fails.

The downloader saves repository files as published on Hugging Face. It does not
load models through Transformers or convert them with `save_pretrained`. The
former `--tensorflow` option is retired and returns a migration error. The old
Docker Compose routes have been removed.

## Offline regression tests

The unit suite requires no Hugging Face, PyTorch, TensorFlow, or model downloads:

```sh
python -m unittest discover -s tests -v
```

CI also installs the pinned official `hfrs` CLI in its temporary runner
directory and downloads only the small public `gpt2/config.json` file to verify
the wrapper-to-CLI integration.
