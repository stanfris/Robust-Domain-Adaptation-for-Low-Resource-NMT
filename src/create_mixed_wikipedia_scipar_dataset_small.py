from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

# Load datasets
wikimedia_dataset = load_from_disk('./data/opus_wikimedia_en_ru')
scipar_dataset = load_dataset('sethjsa/scipar_en_ru_parallel')

# Select 90k samples from Wikimedia
src_lang = "en"
output_lang = "ru"
selected_train_examples = wikimedia_dataset['train'].select(range(0, 9000))
selected_train_dataset = Dataset.from_dict({
    src_lang: [selected_train_examples[i][src_lang] for i in range(len(selected_train_examples))],
    output_lang: [selected_train_examples[i][output_lang] for i in range(len(selected_train_examples))]
})

# Select 10k samples from SciPar
scipar_train = scipar_dataset['train'].select(range(0, 1000))

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
train_dataset.save_to_disk('./data/mixed_dataset_small')


# from datasets import load_from_disk

# # Load the combined dataset from disk
# train_dataset = load_from_disk('../data/mixed_dataset')

# # Select 20 start examples (reproducibly)
# random_examples_1 = train_dataset['train'].select(range(20))

# # select 20 examples from the end of the dataset
# random_examples_2 = train_dataset['train'].select(range(len(train_dataset['train']) - 20, len(train_dataset['train'])))


# # Print the examples
# for i, ex in enumerate(random_examples_1):
#     print(f"Example {i + 1}:")
#     print(f"EN: {ex['en']}")

#     # Print the examples
# for i, ex in enumerate(random_examples_2):
#     print(f"Example {i + 1}:")
#     print(f"EN: {ex['en']}")