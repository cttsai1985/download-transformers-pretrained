# Third-party notices for the container image

This index identifies the top-level third-party components intentionally
included in the runtime image. It is not a complete, per-crate license
inventory of all code incorporated into the `hfrs` executable or its
dependencies.

## hfrs

The image includes the official Hugging Face `hfrs` command-line program,
version 1.0.0, built from [`huggingface/hf-hub` commit
`473de9430b0fbc544fb1d91102180eb713f8dc87`](https://github.com/huggingface/hf-hub/tree/473de9430b0fbc544fb1d91102180eb713f8dc87).
The `hfrs` package declares the Apache License 2.0. Its license text is copied
from the root `LICENSE` file at that source revision and is installed in the
image at:

```text
/usr/share/licenses/download-transformers-pretrained/hfrs/LICENSE
```

The pinned source repository's `Cargo.lock` records the workspace dependency
graph, including packages outside the shipped CLI's runtime dependency set.
This index does not enumerate each linked crate or claim to replace its
individual license and attribution terms. Consult the pinned source and the
license metadata for its dependencies when reviewing those components.

## Python runtime

The image uses the digest-pinned official Python 3.12 slim Bookworm image. Its
Python license file remains at:

```text
/usr/local/lib/python3.12/LICENSE.txt
```

## Debian runtime packages

The Python base image includes Debian Bookworm packages. Their copyright and
license notices remain in the base image under:

```text
/usr/share/doc/<package>/copyright
```

## This project

The project's Apache License 2.0 text is copied into the runtime image at:

```text
/usr/share/licenses/download-transformers-pretrained/LICENSE
```

The license for the Python batch wrapper and repository files is also available
as [`LICENSE`](LICENSE) in the source repository.
