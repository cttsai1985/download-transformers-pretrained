# download-transformers-pretrained

Download Hugging Face repository files with a standalone container. The image
bundles this project's Python batch wrapper and the official Rust `hfrs` CLI.
The host only needs Docker or another compatible container runtime; it does not
need Python, Rust, or `hfrs` installed.

## Get the image

### Load the tested image

The latest successful `master` workflow attaches a private Linux amd64 Docker
image artifact. Open the run in the repository's Actions tab and download the
artifact named `download-transformers-pretrained-linux-amd64-<commit>`. Workflow
artifacts are retained for seven days and require repository access.

```sh
docker load -i download-transformers-pretrained-linux-amd64.tar
```

The imported image is tagged `download-transformers-pretrained:standalone`.
This tested artifact currently targets Linux amd64.

### Build locally

From a checkout, build the image with Docker:

```sh
docker build -t download-transformers-pretrained:standalone .
```

The build fetches the pinned `hfrs` source and uses digest-pinned Rust and
Python base images; you do not need a local Python or Rust toolchain.

## Run the downloader

Create host directories for downloaded files and persistent Hugging Face
client state. The image writes repository files under `/models/<repo-id>` and
uses `/cache` for the official client's cache and authentication state. Mount
both directories to keep their contents after the container exits:

```sh
mkdir -p models hf-cache
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/models" \
  -v "$PWD/hf-cache:/cache" \
  download-transformers-pretrained:standalone \
  --model org/model
```

The image defaults to UID/GID `10001:10001`; the example runs with the current
host user's numeric IDs so files in the bind mount belong to that user. To keep
the image's default identity, make the mounted directories writable by UID/GID
10001. Snapshot files are saved in `/models`; hfrs's `--local-dir` mode reuses
files already there. The separate `/cache` mount preserves the official Hub
client's cache and token state.

To process a repository list, mount it read-only. The default list in the image
is `/app/configs/download.txt`:

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/models" \
  -v "$PWD/hf-cache:/cache" \
  -v "$PWD/configs/download.txt:/config/models.txt:ro" \
  download-transformers-pretrained:standalone \
  --requirement /config/models.txt --include '*.json'
```

The list accepts one repository ID per line; blank lines and lines starting
with `#` are ignored. `--model` selects one repository instead. Repeat
`--include` to pass multiple include patterns. The wrapper attempts every
listed repository and exits nonzero if any download fails. `--output-dir`
overrides the default `/models` directory, and the
`HF_DOWNLOADER_OUTPUT_DIR` environment variable can change that default. The
wrapper delegates authentication, retries, and transfer handling to `hfrs`.

For gated or private repositories, pass the token only when the container runs:

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN \
  -v "$PWD/models:/models" \
  -v "$PWD/hf-cache:/cache" \
  download-transformers-pretrained:standalone \
  --model org/private-model
```

The token is not added to the image. The downloader saves Hub files as
published and does not load or convert models.

## License and bundled components

This repository's wrapper, configuration, and documentation are licensed under
the [Apache License 2.0](LICENSE). The container also bundles the official
`hfrs` 1.0.0 CLI, built from Hugging Face's pinned
[source revision](https://github.com/huggingface/hf-hub/tree/473de9430b0fbc544fb1d91102180eb713f8dc87),
which declares Apache-2.0. The bundled license and component notes are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The Python runtime license
and Debian package copyright notices remain in their original paths in the
image.

The notices file identifies the top-level bundled components; it is not a
complete license inventory of all code incorporated into the `hfrs` executable
or its dependencies. See the pinned upstream source and its `Cargo.lock` for
the dependency graph.

## Offline regression tests

The unit suite requires only Python 3.10+ and performs no Hub or model
downloads:

```sh
python -m unittest discover -s tests -v
```

Maintainers can also run `python script/download.py` directly with the pinned
official `hfrs` CLI on `PATH`; that optional path uses the standard library and
defaults to `./models`.
