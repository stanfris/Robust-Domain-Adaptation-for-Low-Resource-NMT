from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

# Load datasets
tico_dataset = load_dataset('sethjsa/medline_ru_parallel')
scipar_dataset = load_dataset('sethjsa/scipar_en_ru_parallel')

# Select 90k samples from Wikimedia
src_lang = "en"
output_lang = "ru"
selected_train_dataset = tico_dataset['train']

# Select 10k samples from SciPar
scipar_train = scipar_dataset['train']

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
train_dataset.save_to_disk('../data/mixed_dataset_medline_scipar')


