# download-transformers-pretrained
This scripts aims to download transfomers models and configs using docker.

## Usage
To download transformers' models for pytorch:
```docker-compose up```

To download transformers' models for tensorflow:
```docker-compose -f docker-compose-tf.yml up```

Modify `./configs/download.txt` to download different models.

## Output
The default path to save ins at a newly created folder: `./models`
Note it will take up several GB in disk space due to those large models.
