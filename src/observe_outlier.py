import sys
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, FSMTTokenizer

# --- CONFIGURE THESE ---
DATASET_PATH = "../data/mixed_dataset"  # or your dataset path
ROW_INDEX = 10972                     # the problematic row index
MODEL_NAME = "facebook/wmt19-en-ru"     # or your model name
REVERSED_MODEL_NAME = "facebook/wmt19-ru-en"
# -----------------------

def print_tensor_info(name, tensor):
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    else:
        print(f"{name}: type={type(tensor)}")

# Load dataset (from disk or HF)
try:
    dataset = load_from_disk(DATASET_PATH)
    if isinstance(dataset, dict) and "train" in dataset:
        dataset = dataset["train"]
except Exception:
    dataset = load_dataset(DATASET_PATH, split="train")

print(f"Dataset length: {len(dataset)}")
print(f"Inspecting row {ROW_INDEX}:")

# Properly clear the content of the specified row using the map function
def clear_row(example, idx):
    if idx == ROW_INDEX:
        example["en"] = ""
        example["ru"] = ""
    return example

# dataset = dataset.map(clear_row, with_indices=True)

try:
    row = dataset[ROW_INDEX]
    print("Raw row:")
    for k, v in row.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"Error accessing row: {e}")
    sys.exit(1)

try:
    tokenizer = FSMTTokenizer.from_pretrained(MODEL_NAME, langs=["en", "ru"])
    reversed_tokenizer = FSMTTokenizer.from_pretrained(REVERSED_MODEL_NAME, langs=["ru", "en"])
except Exception as e:
    print(f"Error loading tokenizers: {e}")
    sys.exit(1)

inputs = row["en"] if "en" in row else row.get("input", "")
targets = row["ru"] if "ru" in row else row.get("target", "")

print(f"Input text: {inputs}")
print(f"Target text: {targets}")

try:
    # Set language codes for FSMT
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels_tokenized = reversed_tokenizer(text_target = targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels_tokenized["input_ids"]
except Exception as e:
    print(f"Error during tokenization: {e}")
    sys.exit(1)

print("Tokenized model inputs:")
for k, v in model_inputs.items():
    print(f"  {k}: {v}")

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Max input_id: {max(model_inputs['input_ids'])}, Min input_id: {min(model_inputs['input_ids'])}")
print(f"Max label_id: {max(model_inputs['labels'])}, Min label_id: {min(model_inputs['labels'])}")

# Check if any input_ids or labels are >= vocab_size
if max(model_inputs['input_ids']) >= tokenizer.vocab_size:
    print("ERROR: input_ids contain values outside the tokenizer vocab size!")
if max(model_inputs['labels']) >= tokenizer.vocab_size:
    print("ERROR: labels contain values outside the tokenizer vocab size!")

try:
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors="pt")
    train_loader = torch.utils.data.DataLoader(
        [model_inputs], batch_size=1, shuffle=False, collate_fn=data_collator)
except Exception as e:
    print(f"Error creating DataLoader: {e}")
    sys.exit(1)

try:
    from transformers import AutoModelForSeq2SeqLM
    import torch

    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.cuda()
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

try:
    for batch in train_loader:
        print("Batch keys:", list(batch.keys()))
        for k, v in batch.items():
            print_tensor_info(k, v)
        model_inputs = {k: v.cuda() for k, v in batch.items()}
        print("Moved batch to CUDA.")
        for k, v in model_inputs.items():
            print_tensor_info(f"{k} (cuda)", v)
        print("Running forward pass...")
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**model_inputs)
        print("Forward pass successful. Outputs:", outputs)
        print("Running backward pass...")
        outputs.loss.backward()
        print("Backward pass successful.")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm()}")
            else:
                print(f"No gradient for {name}")
        print("Model ran successfully without errors.")
except RuntimeError as e:
    import traceback
    print("CUDA/Runtime error during model execution:")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    import traceback
    print("General error during model execution:")
    traceback.print_exc()
    sys.exit(1)
