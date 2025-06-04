from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import random

# Load datasets
tico_dataset = load_dataset('sethjsa/tico_en_ru')
scipar_dataset = load_dataset('sethjsa/scipar_en_ru_parallel')

# Select 90k samples from Wikimedia
src_lang = "en"
output_lang = "ru"
selected_train_dataset = tico_dataset['test']

# Select 10k samples from SciPar
scipar_train = scipar_dataset['train']

# corrupt the scipar dataset
def corrupt_text(text):
    # replace random characters inside words with random characters
    words = text.split()
    corrupted_words = []
    for word in words:
        # replace word with random characters
        word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') if random.random() < 0.5 else char for char in word)
        corrupted_words.append(word)
        # delete word with 30% probability
        if random.random() < 0.3:
            corrupted_words.pop()
    
    return ' '.join(corrupted_words)

        


scipar_train = scipar_train.map(lambda x: {src_lang: corrupt_text(x[src_lang]), output_lang: corrupt_text(x[output_lang])})
# Combine into one training set
combined_train_dataset = Dataset.from_dict({
    src_lang: [ex[src_lang] for ex in selected_train_dataset] + [ex[src_lang] for ex in scipar_train],
    output_lang: [ex[output_lang] for ex in selected_train_dataset] + [ex[output_lang] for ex in scipar_train]
})

# Create the final DatasetDict
train_dataset = DatasetDict({
    'train': combined_train_dataset
})

# Save to disk
train_dataset.save_to_disk('../data/corrupted_mixed_scipar_tico')


