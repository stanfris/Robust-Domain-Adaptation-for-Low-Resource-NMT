from modules import *
from transformers import FSMTForConditionalGeneration
# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from datasets import load_dataset, Dataset, DatasetDict
# from evaluate import load
# import numpy as np
# # import vllm
# from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Translation Model Training/Fine-Tuning')

# Language and model
parser.add_argument('--src_lang', default='ru', type=str,
                    help='Source language code')
parser.add_argument('--tgt_lang', default='en', type=str,
                    help='Target language code')
parser.add_argument('--model_name', required=True, type=str,
                    help='Pretrained model name')

parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

# Dataset configuration
parser.add_argument('--dataset', default='sethjsa/medline_ru_mono', type=str,
                    help='Name of the test dataset')

parser.add_argument('--output_dir', required=True, type=str, help='Where to save the dataset')


if __name__ == "__main__":
    args = parser.parse_args()

    # Check if CUDA is available and not disabled
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected! Training will be slow.")
        raise RuntimeError("No GPU detected! Training will be slow.")
    
    train_dataset = load_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = FSMTForConditionalGeneration.from_pretrained(args.model_name)

    print(model.state_dict())

    for name, param in model.named_parameters():
        if param == 'model.decoder.output_projection.weight':
            print(f'qwq! {param.device}')
        if param.device == torch.device("meta"):
            print(f"Meta tensor found in module: {name}")

    # # test model
    russian_texts = train_dataset["train"][args.src_lang]
    english_texts = translate_text(russian_texts, model, tokenizer, max_length=512, batch_size=64)

    # Create new Dataset with RU and EN columns
    new_dataset = Dataset.from_dict({"ru": russian_texts, "en": english_texts})
    new_dataset.save_to_disk(args.output_dir)

    print(f"\n✅ Backtranslated dataset saved to: {args.output_dir}")
    print(new_dataset[:3])  # Preview a few samples