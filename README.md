# download-transformers-pretrained
This scripts aims to download transfomers models and configs using docker.

## Usage
To download transformers' models for pytorch:
```docker-compose up```

To download transformers' models for tensorflow:
```docker-compose -f docker-compose-tf.yml up```

Modify `./configs/download.txt` to download different models.

With PyTorch and Transformers installed, you can also run the downloader directly:

```sh
python script/download.py
python script/download.py --model org/model --output-dir ./models
```

The default model list and output path are resolved relative to this repository,
so direct execution also works from another working directory. Namespaced model
IDs create matching nested directories. A failed download or save exits nonzero.
TensorFlow mode requires a Transformers version that provides `TFAutoModel` and
a compatible TensorFlow installation; PyTorch mode does not import this class.

Run the offline regression suite without installing Transformers or downloading models:

```sh
python -m unittest discover -s tests -v
```

## Output
The default path to save ins at a newly created folder: `./models`
Note it will take up several GB in disk space due to those large models.
