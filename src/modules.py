import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from evaluate import load
import numpy as np
# import vllm
from tqdm import tqdm
import random
import copy
from rank_bm25 import BM25Okapi
import re 
from collections import Counter
# PREPROCESSING
# function to tokenize dataset for translation tasks
def preprocess_data(dataset_dict, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, split, max_length=128):
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
        print(targets[:8])

        model_inputs = tokenizer_src(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        labels = tokenizer_tgt(
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


# function to augment data for backtranslation following Edunov et al. (2018)
def augment_text(text, p):
    words = text.split()
    index = 0

    while index < len(words):
        r = random.random()
        if r < p:
            words.pop(index)

        elif r < (2*p):
            words[index] = "<mask>"
            index += 1

        elif r < (3*p):
            left = max(0, index - 3)
            right = min(len(words) - 1, index + 3)
            partners = [j for j in range(left, right + 1) if j != index]
            if partners:
                swap_with = random.choice(partners)
                words[index], words[swap_with] = words[swap_with], words[index]
            index += 1

        else:
            index += 1

    return " ".join(words)

def augment_dataset(dataset, p=0.1):
    return [augment_text(text, p) for text in dataset]

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


# EVALUATION

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

    print(decoded_preds[0:8], decoded_labels[0:8])

    # Calculate BLEU score
    bleu = load("sacrebleu")
    results = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    return {"bleu": results["score"]}

def compute_metrics_test(src, tgt, preds, tokenizer, bleu=True, comet=False, bert_score=False):
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
    # detokenize 

    metrics = {}

    if bleu:
        bleu = load("sacrebleu")
        results = bleu.compute(predictions=preds, references=[[l] for l in tgt])
        metrics["bleu"] = results["score"]

    if comet:
        comet = load("comet")
        results = comet.compute(predictions=preds, references=tgt, sources=src)
        metrics["comet"] = results["mean_score"]
    
    if bert_score:
        bert_score = load("bertscore")
        results = bert_score.compute(predictions=preds, references=tgt, lang='ru')
        metrics["bert_score_f1"] = np.mean(results["f1"])

    return metrics



# TRAINING
# basic training loop

def train_model(model, tokenized_datasets, tokenizer, training_args, train_split="train", dev_split="dev"):
    print("Training model...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[train_split],
        eval_dataset=tokenized_datasets[dev_split] if dev_split in tokenized_datasets else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda x: compute_metrics_val(tokenizer, x)
    )

    trainer.train()
    print("Training complete!")
    return model


def less_data_selection(model, tokenized_datasets, tokenizer, training_args):
    return None

def tokenize_bm25(text):
    """ Tokenizes text into a list of words, lowercase and word-boundaries. """
    return re.findall(r"\b\w+\b", text.lower())

def extract_ngrams(tokens, n=4):
    """Return a set of word‐level n‑grams (as tuples)."""
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def select_data_subset(model, train_dataset, dev_dataset, tokenized_dev_set, dev_sample_percentage,save_percentage, tokenizer, training_args, train_split="train", dev_split="dev", selection_method="bm25", src_lang="en", output_lang="ru"):
    """
    Selects a subset of the training data based on BM25 scores relative to the dev queries.

    Args:
        train_dataset: The training dataset (dict-like, should contain 'train' and 'en' keys).
        dev_dataset: The development dataset (dict-like, should contain 'dev' and 'en' keys).
        k: The number of top examples to select based on BM25 scores.

    Returns:
        Updated train_dataset with selected subset of examples.
    """
    if selection_method == "bm25":
        train_corpus = [tokenize_bm25(doc) for doc in train_dataset[train_split][src_lang]]

        dev_size = int(dev_sample_percentage * len(dev_dataset[dev_split]))
        dev_corpus = [tokenize_bm25(doc) for doc in dev_dataset[dev_split].shuffle().select(range(dev_size))[src_lang]]

        bm25 = BM25Okapi(train_corpus)

        all_scores = [bm25.get_scores(dev_doc) for dev_doc in dev_corpus]
        score_matrix = np.stack(all_scores)      # shape (n_dev, n_train)

        final_scores = score_matrix.sum(axis=0)  # shape (n_train,)

        k = int(len(final_scores) * save_percentage)
        top_k_indices = np.argsort(-final_scores)[:k]

        selected_train_examples = train_dataset[train_split].select(top_k_indices)
        selected_train_dataset = Dataset.from_dict({
            output_lang: [selected_train_examples[i][output_lang] for i in range(len(selected_train_examples))],
            src_lang: [selected_train_examples[i][src_lang] for i in range(len(selected_train_examples))]

        })

        new_dataset_dict = DatasetDict({
            train_split: selected_train_dataset,
        })



    elif selection_method == "5gram":
        # Tokenize to word lists
        train_texts = train_dataset[train_split][src_lang]
        dev_texts   = dev_dataset[dev_split][src_lang]
        train_tokens = [txt.split() for txt in train_texts]
        dev_tokens   = [txt.split() for txt in dev_texts]

        dev_counter = Counter()
        for toks in dev_tokens:
            dev_counter.update(extract_ngrams(toks, n=5))

        final_scores = np.array([
            sum(dev_counter[gram] for gram in extract_ngrams(toks, n=5))
            for toks in train_tokens
        ])

        k = int(len(final_scores) * save_percentage)
        top_k_indices = np.argsort(-final_scores)[:k]

        selected_train_examples = train_dataset[train_split].select(top_k_indices)
        selected_train_dataset = Dataset.from_dict({
            output_lang: [selected_train_examples[i][output_lang] for i in range(len(selected_train_examples))],
            src_lang: [selected_train_examples[i][src_lang] for i in range(len(selected_train_examples))]
        })
        new_dataset_dict = DatasetDict({
            train_split: selected_train_dataset,
        })



    elif selection_method == "random":
        # Randomly select a percentage of the training dataset
        num_samples = int(len(train_dataset[train_split]) * save_percentage)
        indices = random.sample(range(len(train_dataset[train_split])), num_samples)
        selected_train_examples = train_dataset[train_split].select(indices)

        selected_train_dataset = Dataset.from_dict({
            output_lang: [selected_train_examples[i][output_lang] for i in range(len(selected_train_examples))],
            src_lang: [selected_train_examples[i][src_lang] for i in range(len(selected_train_examples))]
        })

        new_dataset_dict = DatasetDict({
            train_split: selected_train_dataset,
        })
    
    if selection_method == "LESS" :
        dev_size = int(dev_sample_percentage * len(tokenized_dev_set[dev_split]))
        warmup_dev = tokenized_dev_set[dev_split].shuffle().select(range(dev_size))

        model_warmup = copy.deepcopy(model)

        # train model on 5% of dev set
        print("Warmup, training model on 5% of dev set...")
        warmup_trainer = Seq2SeqTrainer(
            model=model_warmup,
            args=training_args,
            train_dataset=warmup_dev,
            eval_dataset=tokenized_dev_set.get(dev_split, None),
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
        )
        warmup_trainer.train()

        new_train_dataset = less_data_selection(model_warmup, train_dataset, warmup_dev, tokenizer, training_args)
        new_dataset_dict = DatasetDict({
            train_split: new_train_dataset,
        })

    return new_dataset_dict


        

# GENERATION 

# generation (on GPU) for test time
def translate_text(texts, model, tokenizer, max_length=128, batch_size=4, temperature=0.0):
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
                temperature=temperature,
                do_sample=temperature >0,
                early_stopping=True
            )

        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)

    return translations