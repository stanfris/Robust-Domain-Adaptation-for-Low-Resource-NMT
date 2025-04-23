# imports
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict
from evaluate import load
import numpy as np
# import vllm
from tqdm import tqdm


# function to tokenize dataset for translation

def preprocess_data(dataset_dict, tokenizer, src_lang, tgt_lang, split, max_length=128):
    """
    Preprocess translation datasets

    Args:
        dataset_dict: Dictionary containing train/dev/test datasets
        tokenizer: Tokenizer object
        src_lang: Source language code
        tgt_lang: Target language code
        split: Dataset split to preprocess ('train', 'validation', etc)
        max_length: Maximum sequence length
    Returns:
        tokenized_dataset: Preprocessed dataset for specified split
    """
    def preprocess_function(examples):
        inputs = examples[src_lang]
        targets = examples[tgt_lang]

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset_dict[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict[split].column_names
    )

    return tokenized_dataset

def postprocess_predictions(predictions, labels, tokenizer):
    """
    Convert model outputs to decoded text

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        tokenizer: Tokenizer object
    Returns:
        decoded_preds: Decoded predictions
        decoded_labels: Decoded labels
    """
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return decoded_preds, decoded_labels

# evaluation: for validation (with raw outputs) and testing (from text)

def compute_metrics_val(tokenizer, eval_preds):
    """
    Calculate BLEU score for predictions

    Args:
        tokenizer: Tokenizer object
        eval_preds: Tuple of predictions and labels
    Returns:
        metrics: Dictionary containing BLEU score
    """
    preds, labels = eval_preds
    decoded_preds, decoded_labels = postprocess_predictions(preds, labels, tokenizer)

    # Calculate BLEU score
    bleu = load("sacrebleu")
    results = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    return {"bleu": results["score"]}

def compute_metrics_test(src, tgt, preds, bleu=True, comet=False):
    """
    Calculate BLEU score for predictions

    Args:
        src: Source language texts
        tgt: Target language texts
        preds: Predicted texts
        bleu: Whether to calculate BLEU score
        comet: Whether to calculate COMET score
    Returns:
        metrics: Dictionary containing BLEU score
    """
    if bleu:
        bleu = load("sacrebleu")
        results = bleu.compute(predictions=preds, references=[[l] for l in tgt])
        score = results["score"]
    if comet:
      raise NotImplementedError("COMET not implemented yet")
        # Calculate COMET score

    return score

# basic training loop

def train_model(model_name, tokenized_datasets, tokenizer, training_args):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Verify GPU usage
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! Training will be slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"] if "dev" in tokenized_datasets else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda x: compute_metrics_val(tokenizer, x)
    )

    trainer.train()
    return model

# generation (on GPU) for test time
def translate_text(texts, model, tokenizer, max_length=128, batch_size=32):
    """
    Translate texts using the model

    Args:
        texts: List of texts to translate
        model: Translation model
        tokenizer: Tokenizer object
        max_length: Maximum sequence length
        batch_size: Batch size for translation
    Returns:
        translations: List of translated texts
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    translations = []

    # Create tqdm progress bar
    progress_bar = tqdm(range(0, len(texts), batch_size), desc="Translating")

    for i in progress_bar:
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.0,
                early_stopping=True
            )

        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)

    return translations

SRC_LANG = "en"
TGT_LANG = "ru"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
TRAIN_DATASET_NAME = "sethjsa/flores_en_ru"
DEV_DATASET_NAME = "sethjsa/tico_en_ru"
TEST_DATASET_NAME = "sethjsa/tico_en_ru"
OUTPUT_DIR = "./results"

train_dataset = load_dataset(TRAIN_DATASET_NAME)
dev_dataset = load_dataset(DEV_DATASET_NAME)
test_dataset = load_dataset(TEST_DATASET_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# change the splits for actual training. here, using flores-dev as training set because it's small (<1k examples)
tokenized_train_dataset = preprocess_data(train_dataset, tokenizer, SRC_LANG, TGT_LANG, "dev")
tokenized_dev_dataset = preprocess_data(dev_dataset, tokenizer, SRC_LANG, TGT_LANG, "dev")
tokenized_test_dataset = preprocess_data(test_dataset, tokenizer, SRC_LANG, TGT_LANG, "test")

tokenized_datasets = DatasetDict({
    "train": tokenized_train_dataset,
    "dev": tokenized_dev_dataset,
    "test": tokenized_test_dataset
})

# modify these as you wish; RQ3 could involve testing effects of various hyperparameters
training_args = Seq2SeqTrainingArguments(
    torch_compile=True, # generally speeds up training, try without it to see if it's faster for small datasets
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32, # change batch sizes to fit your GPU memory and train faster
    per_device_eval_batch_size=128,
    weight_decay=0.01,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    save_total_limit=1, # modify this to save more checkpoints
    num_train_epochs=1, # modify this to train more epochs
    predict_with_generate=True,
    generation_num_beams=4,
    generation_max_length=128,
    no_cuda=False,  # Set to False to enable GPU
    fp16=True,      # Enable mixed precision training for faster training
)


# fine-tune model
model = train_model(MODEL_NAME, tokenized_datasets, tokenizer, training_args)

# test model
predictions = translate_text(test_dataset["test"][SRC_LANG], model, tokenizer, max_length=128, batch_size=64)
print(predictions)

eval_score = compute_metrics_test(test_dataset["test"][SRC_LANG], test_dataset["test"][TGT_LANG], predictions, bleu=False, comet=True)
print(eval_score)