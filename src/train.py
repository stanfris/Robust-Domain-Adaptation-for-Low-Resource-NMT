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
parser.add_argument('--model_name', default='facebook/wmt19-en-ru', type=str,
                    help='Pretrained model name')

# Dataset configuration
parser.add_argument('--train_dataset', default='sethjsa/flores_en_ru', type=str,
                    help='Name of the training dataset')
parser.add_argument('--dev_dataset', default='sethjsa/tico_en_ru', type=str,
                    help='Name of the development/validation dataset')
parser.add_argument('--test_dataset', default='sethjsa/tico_en_ru', type=str,
                    help='Name of the test dataset')

# Preprocessing
parser.add_argument('--train_split', default='dev', type=str,
                    help='Split to use for training (e.g., train, dev)')
parser.add_argument('--dev_split', default='dev', type=str,
                    help='Split to use for dev')
parser.add_argument('--test_split', default='test', type=str,
                    help='Split to use for test')

# Output
parser.add_argument('--output_dir', default='../results', type=str,
                    help='Directory to save checkpoints and logs')

# Training arguments
parser.add_argument('--num_train_epochs', default=1, type=int,
                    help='Number of training epochs')
parser.add_argument('--learning_rate', default=2e-5, type=float,
                    metavar='LR', help='Initial learning rate')
parser.add_argument('--per_device_train_batch_size', default=32, type=int,
                    help='Training batch size per device')
parser.add_argument('--per_device_eval_batch_size', default=128, type=int,
                    help='Evaluation batch size per device')
parser.add_argument('--weight_decay', default=0.01, type=float,
                    metavar='W', help='Weight decay (L2 regularization)')
parser.add_argument('--adam_beta1', default=0.9, type=float,
                    help='Beta1 for Adam optimizer')
parser.add_argument('--adam_beta2', default=0.999, type=float,
                    help='Beta2 for Adam optimizer')
parser.add_argument('--adam_epsilon', default=1e-8, type=float,
                    help='Epsilon for Adam optimizer')
parser.add_argument('--optim', default='adamw_torch', type=str,
                    help='Optimizer to use')
parser.add_argument('--save_total_limit', default=1, type=int,
                    help='Limit the total amount of checkpoints. Deletes the older checkpoints.')
parser.add_argument('--evaluation_strategy', default='epoch', type=str,
                    help='Evaluation strategy (e.g., "epoch", "steps")')
parser.add_argument('--predict_with_generate', action='store_true',
                    help='Use generate() to calculate generative metrics')
parser.add_argument('--generation_num_beams', default=4, type=int,
                    help='Number of beams for beam search during generation')
parser.add_argument('--generation_max_length', default=128, type=int,
                    help='Maximum length for generation')
parser.add_argument('--no_cuda', action='store_true',
                    help='Disable CUDA even if available')
parser.add_argument('--fp16', action='store_true',
                    help='Use mixed precision training (FP16)')
parser.add_argument('--torch_compile', action='store_true',
                    help='Use torch.compile() for model acceleration')


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
    

    train_dataset = load_dataset(args.train_dataset)
    dev_dataset = load_dataset(args.dev_dataset)
    test_dataset = load_dataset(args.test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # change the splits for actual training. here, using flores-dev as training set because it's small (<1k examples)
    tokenized_train_dataset = preprocess_data(train_dataset, tokenizer, args.src_lang, args.tgt_lang, "dev")
    tokenized_dev_dataset = preprocess_data(dev_dataset, tokenizer, args.src_lang, args.tgt_lang, "dev")
    tokenized_test_dataset = preprocess_data(test_dataset, tokenizer, args.src_lang, args.tgt_lang, "test")

    tokenized_datasets = DatasetDict({
        "train": tokenized_train_dataset,
        "dev": tokenized_dev_dataset,
        "test": tokenized_test_dataset
    })

    # modify these as you wish; RQ3 could involve testing effects of various hyperparameters

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=args.predict_with_generate,
        generation_num_beams=args.generation_num_beams,
        generation_max_length=args.generation_max_length,
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        torch_compile=args.torch_compile,
    )

    # fine-tune model
    model = train_model(args.model_name, tokenized_datasets, tokenizer, training_args)

    # test model
    predictions = translate_text(test_dataset["test"][args.src_lang], model, tokenizer, max_length=128, batch_size=64)
    print(predictions)

    eval_score = compute_metrics_test(test_dataset["test"][args.src_lang], test_dataset["test"][args.tgt_lang], predictions, bleu=False, comet=True)
    print(eval_score)