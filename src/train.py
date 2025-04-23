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
parser.add_argument('--src_lang', default='en', type=str, help='Source language code')
parser.add_argument('--tgt_lang', default='ru', type=str, help='Target language code')
parser.add_argument('--model_name', default='Helsinki-NLP/opus-mt-en-ru', type=str, help='Pretrained model name')

# Dataset configuration
parser.add_argument('--train_dataset', default='sethjsa/flores_en_ru', type=str, help='Name of the training dataset')
parser.add_argument('--dev_dataset', default='sethjsa/tico_en_ru', type=str, help='Name of the development dataset')
parser.add_argument('--test_dataset', default='sethjsa/tico_en_ru', type=str, help='Name of the test dataset')
parser.add_argument('--train_split', default='dev', type=str, help='Dataset split to use for training')
parser.add_argument('--dev_split', default='dev', type=str, help='Dataset split to use for development')
parser.add_argument('--test_split', default='test', type=str, help='Dataset split to use for testing')

# Output and logging
parser.add_argument('--output_dir', default='../results', type=str, help='Directory to save outputs')
parser.add_argument('--report_to', default='tensorboard', type=str, help='Reporting tool to use (e.g., tensorboard)')

# Training configuration
parser.add_argument('--num_train_epochs', default=1, type=int, help='Number of training epochs')
parser.add_argument('--learning_rate', default=2e-5, type=float, help='Initial learning rate')
parser.add_argument('--per_device_train_batch_size', default=32, type=int, help='Training batch size per device')
parser.add_argument('--per_device_eval_batch_size', default=128, type=int, help='Evaluation batch size per device')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
parser.add_argument('--optim', default='adamw_torch', type=str, help='Optimizer to use')
parser.add_argument('--adam_beta1', default=0.9, type=float, help='Beta1 for Adam optimizer')
parser.add_argument('--adam_beta2', default=0.999, type=float, help='Beta2 for Adam optimizer')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer')
parser.add_argument('--save_total_limit', default=1, type=int, help='Max number of saved checkpoints')
parser.add_argument('--save_strategy', default='epoch', type=str, help='Checkpoint saving strategy')
parser.add_argument('--eval_strategy', default='epoch', type=str, help='Evaluation strategy')
parser.add_argument('--predict_with_generate', action='store_true', help='Use generate() for predictions')
parser.add_argument('--generation_num_beams', default=4, type=int, help='Number of beams for generation')
parser.add_argument('--generation_max_length', default=128, type=int, help='Maximum sequence length for generation')
parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--fp16', action='store_false', help='Enable mixed precision training')
parser.add_argument('--torch_compile', action='store_true', help='Enable torch.compile() for acceleration')
parser.add_argument('--load_best_model_at_end', action='store_true', help='Load best model after training ends')
parser.add_argument('--metric_for_best_model', default='bleu', type=str, help='Metric to determine best model')
parser.add_argument('--greater_is_better', action='store_true', help='If greater is better for the selected metric')


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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

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
        evaluation_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
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
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to=args.report_to,
    )

    training_args = Seq2SeqTrainingArguments(
        torch_compile=True, # generally speeds up training, try without it to see if it's faster for small datasets
    output_dir=args.output_dir,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32, # change batch sizes to fit your GPU memory and train faster
    per_device_eval_batch_size=128,
    weight_decay=0.01,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    save_total_limit=10, # modify this to save more checkpoints
    num_train_epochs=10, # modify this to train more epochs
    predict_with_generate=True,
    generation_num_beams=4,
    generation_max_length=128,
    no_cuda=False,  # Set to False to enable GPU
    fp16=True,      # Enable mixed precision training for faster training
    load_best_model_at_end=True, # load the best model at the end of training
    metric_for_best_model="bleu", # metric to use for best model selection
    greater_is_better=True, # whether higher or lower is better for the metric
    save_strategy="epoch", # save model every epochpip install tensorboardX
    report_to="tensorboard", # log to tensorboard
        )  
    # training model
    # fine-tune model
    print("Training model...")
    model = train_model(model, tokenized_datasets, tokenizer, training_args)

    # # save model 
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
