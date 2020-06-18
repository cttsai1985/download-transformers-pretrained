import os
import argparse
from typing import Union
from pathlib import Path

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel


def mkdir(dir_path: Union[str, Path]) -> bool:
    if os.path.isdir(dir_path):
        print(f"Skip created directory: {dir_path}")
        return True

    try:
        os.mkdir(dir_path)
        print(f"Successfully created directory: {dir_path}")

    except OSError:
        print(f"Creation of directory: {dir_path} failed")
        return False

    return True


def transformers_dowloader(
        pretrained_model_name: str, working_dir: Union[str, Path], is_tf: bool = True) -> bool:
    model_class = AutoModel
    if is_tf:
        model_class = TFAutoModel

    print(f"Download model and tokenizer for: {pretrained_model_name}")
    transformer_model = model_class.from_pretrained(pretrained_model_name)
    transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    #

    output_dir = working_dir / pretrained_model_name
    try:
        mkdir(dir_path=output_dir)
        transformer_model.save_pretrained(output_dir)
        transformer_tokenizer.save_pretrained(output_dir)
        print(f"Save model and tokenizer {pretrained_model_name} in directory {output_dir}")

    except OSError:
        print(f"Save model and tokenizer {pretrained_model_name} in directory {output_dir}: Failed")
        return False

    return True


def main():
    default_output_dir: str = "../models"
    default_data_filename: str = "models_to_download.txt"

    parser = argparse.ArgumentParser(
        description="Transformers Pretrained Models Downloader", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="folder to save")
    parser.add_argument(
        "--requirement", "-r", type=str, default=default_data_filename, help="file of model to download")
    parser.add_argument("--model", "-m", type=str, default=None, help="filename of data")
    parser.add_argument("--tensorflow", action="store_true", default=False, help="cross val group split")
    args = parser.parse_args()

    pretrained_model_name_list = [args.model]
    if args.model is None:
        with open(args.requirement, 'r') as opened_file:
            pretrained_model_name_list = [s.strip("\n").strip(" ") for s in opened_file.readlines()]
            pretrained_model_name_list = list(filter(lambda s: not s.startswith("#"), pretrained_model_name_list))
            pretrained_model_name_list = list(filter(lambda s: len(s) > 0, pretrained_model_name_list))
            print(f"Plan: download {len(pretrained_model_name_list)} models: {', '.join(pretrained_model_name_list)}")

    # 'bert-base-uncased'

    print(f'Transformers version {transformers.__version__}')
    working_dir = Path(args.output_dir)
    mkdir(working_dir)

    for model_name in pretrained_model_name_list:
        transformers_dowloader(model_name, working_dir=working_dir, is_tf=args.tensorflow)
    return


if "__main__" == __name__:
    main()
