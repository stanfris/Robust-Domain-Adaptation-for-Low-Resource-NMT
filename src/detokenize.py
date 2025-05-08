from transformers import AutoTokenizer, FSMTTokenizer

# Load the tokenizer
tokenizer_en_ru = AutoTokenizer.from_pretrained('facebook/wmt19-en-ru')
tokenizer_ru_en = AutoTokenizer.from_pretrained('facebook/wmt19-ru-en')

# Example input text (English) and target text (Russian)
input_text = ["How long have these symptoms been present?"]
target_text = ["о том, как долго присутствуют эти симптомы?"]

# Tokenize the input text (English)
inputs = tokenizer_en_ru(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Tokenize the target text (Russian)
labels = tokenizer_ru_en(target_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Ensure labels are correctly aligned and that padding tokens are not considered during training
labels["input_ids"][labels["input_ids"] == tokenizer_ru_en.pad_token_id] = -100  # Set padding tokens to -100

# Decode the target tokens to inspect
decoded_target = tokenizer_en_ru.decode(labels["input_ids"][0], skip_special_tokens=True)

# Print decoded target tokens to see the result
print(f"Decoded Target: {decoded_target}")
