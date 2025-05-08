from modules import *
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
parser.add_argument('--src_lang', default='en', type=str,
                    help='Source language code')
parser.add_argument('--tgt_lang', default='ru', type=str,
                    help='Target language code')
parser.add_argument('--model_name', default='Helsinki-NLP/opus-mt-en-ru', type=str,
                    help='Pretrained model name')
parser.add_argument('--reversed_model_name', default=None, type=str,
                    help='Pretrained tokenizer name for encoding target language')

parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

# Dataset configuration
parser.add_argument('--test_dataset', default='sethjsa/tico_en_ru', type=str,
                    help='Name of the test dataset')


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    # Check if CUDA is available and not disabled
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected! Training will be slow.")
        raise RuntimeError("No GPU detected! Training will be slow.")
    
    test_dataset = load_dataset(args.test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.reversed_model_name is None:
        args.reversed_model_name = args.model_name
    tokenizer_tgt = AutoTokenizer.from_pretrained(args.reversed_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    # change the splits for actual training. here, using flores-dev as training set because it's small (<1k examples)
    tokenized_test_dataset = preprocess_data(test_dataset, tokenizer, tokenizer_tgt, args.src_lang, args.tgt_lang, "test")

    tokenized_datasets = DatasetDict({
        "test": tokenized_test_dataset
    })


    # # test model
    predictions = translate_text(test_dataset["test"][args.src_lang], model, tokenizer, max_length=512, batch_size=64)

    eval_score = compute_metrics_test(test_dataset["test"][args.src_lang], test_dataset["test"][args.tgt_lang], predictions, bleu=True, comet=True, bert_score=True, tokenizer=tokenizer)
    print('evaluation score:', eval_score)