# download-transformers-pretrained

Download Hugging Face repository files in a standalone container. The image
bundles the batch wrapper and the official Rust `hfrs` CLI; the host needs a
container runtime, with no local Python, Rust toolchain, or `hfrs` installation.

## Run the container

Create directories for downloads and the persistent Hugging Face cache. The
image defaults to `/models` for output and `/cache` for Hugging Face cache and
authentication state. Mount both outside the container so files and cache
survive container removal:

```sh
mkdir -p models hf-cache
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/models" \
  -v "$PWD/hf-cache:/cache" \
  download-transformers-pretrained:standalone \
  --model org/model
```

The image itself runs as UID/GID `10001:10001`. The example overrides that
identity with the current host user's numeric IDs so bind-mounted output is
writable and owned by that user. Create the host directories first. To use the
image's default identity instead, make the mounted directories writable by
UID/GID 10001.

Pass repeated include filters or a read-only list file as needed:

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/models" \
  -v "$PWD/hf-cache:/cache" \
  -v "$PWD/configs/download.txt:/config/models.txt:ro" \
  download-transformers-pretrained:standalone \
  --requirement /config/models.txt --include '*.json'
```

For gated or private repositories, provide the token only when the container
runs. Avoid putting it in a Dockerfile, image, command line, or build argument:

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN \
  -v "$PWD/models:/models" \
  -v "$PWD/hf-cache:/cache" \
  download-transformers-pretrained:standalone \
  --model org/private-model
```

The wrapper saves raw Hub repository files under `/models/<repo-id>` and leaves
authentication, cache handling, retries, and file selection to `hfrs`. Its
output default can be changed with `HF_DOWNLOADER_OUTPUT_DIR`; `--output-dir`
overrides the environment value. The list file accepts one repository ID per
line; blank lines and lines starting with `#` are ignored. The wrapper attempts
every listed repository and exits nonzero if any download fails. The retired
`--tensorflow` option returns a migration error; no model loading or format
conversion is performed.

## Build locally or load the tested image

Build the image from this repository:

```sh
docker build -t download-transformers-pretrained:standalone .
```

The workflow builds and exercises the Linux amd64 image on every pull request
and on master pushes. It attaches the tested Docker image as a private workflow
artifact with seven-day retention. Download the artifact tar and load it
without building locally:

```sh
docker load -i download-transformers-pretrained-linux-amd64.tar
```

The image contains the Python standard library, the batch wrapper, and pinned
official `hfrs` 1.0.0. It does not include Python ML frameworks, a Rust
toolchain, model weights, credentials, or host cache files.

For maintainers running the wrapper directly, put the same pinned official
`hfrs` CLI on `PATH` and run `python script/download.py`; this optional path
uses the standard library and keeps the native `./models` output default.

## Offline regression tests

The unit suite requires only Python 3.10+ and performs no Hub or model
downloads:

```sh
python -m unittest discover -s tests -v
```
