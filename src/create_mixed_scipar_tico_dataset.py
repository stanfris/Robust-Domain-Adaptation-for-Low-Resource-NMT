# from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

# # Load datasets
# tico_dataset = load_dataset('sethjsa/tico_en_ru')
# scipar_dataset = load_dataset('sethjsa/scipar_en_ru_parallel')

# # Select 90k samples from Wikimedia
# src_lang = "en"
# output_lang = "ru"
# selected_train_dataset = tico_dataset['test']

# # Select 10k samples from SciPar
# scipar_train = scipar_dataset['train']

# # Combine into one training set
# combined_train_dataset = Dataset.from_dict({
#     src_lang: [ex[src_lang] for ex in selected_train_dataset] + [ex[src_lang] for ex in scipar_train],
#     output_lang: [ex[output_lang] for ex in selected_train_dataset] + [ex[output_lang] for ex in scipar_train]
# })

# # Create the final DatasetDict
# train_dataset = DatasetDict({
#     'train': combined_train_dataset
# })

# # Save to disk
# train_dataset.save_to_disk('../data/mixed_dataset_tico_scipar')


from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from modules import *

# Load the combined dataset from disk
train_dataset = load_from_disk('data/mixed_dataset')
dev_dataset = load_dataset('sethjsa/medline_ru_parallel')

# Select 20 start examples (reproducibly)
# random_examples_1 = train_dataset['train'].select(range(20))

# # select 20 examples from the end of the dataset
# random_examples_2 = train_dataset['train'].select(range(len(train_dataset['train']) - 20, len(train_dataset['train'])))

model, tokenized_dev_dataset, tokenizer, training_args = 1, 1, 1, 1
dev_sample_percentage = 0.05
save_percentage = 0.05
train_split="train"
dev_split="train"
# print length of train_dataset
print(f"Length of train_dataset: {len(train_dataset['train'])}")
print("Loading model...")	
selected_data, indices = select_data_subset(model, train_dataset, dev_dataset, tokenized_dev_dataset, dev_sample_percentage, save_percentage, tokenizer, training_args, train_split=train_split, dev_split=dev_split, selection_method="bm25", src_lang="en", output_lang="ru", return_indices=True)

# print what percentage of indices are below 2100
print(f"Percentage of indices below 2100 bm25: {sum([1 for i in indices if i > 90000]) / len(indices) * 100:.2f}%")
# # print the indexed first 10 indices
# indexed_train_dataset = train_dataset['train'].select(indices[:5])['en']
# print("Indexed train dataset:")
# for i, ex in enumerate(indexed_train_dataset):
#     print(f"Example {i + 1}:")
#     print(ex)
# print("Selected data:")
# for i, ex in enumerate(selected_data['train']['en'][:5]):
#     print(f"Example {i + 1}:")
#     print(ex)
# selected_data, indices = select_data_subset(model, train_dataset, dev_dataset, tokenized_dev_dataset, dev_sample_percentage, save_percentage, tokenizer, training_args, train_split=train_split, dev_split=dev_split, selection_method="5gram", src_lang="en", output_lang="ru", return_indices=True)
# print(f"Percentage of indices below 2100 5gram: {sum([1 for i in indices if i > 10000]) / len(indices) * 100:.2f}%")


selected_data, indices = select_data_subset(model, train_dataset, dev_dataset, tokenized_dev_dataset, dev_sample_percentage, save_percentage, tokenizer, training_args, train_split=train_split, dev_split=dev_split, selection_method="random", src_lang="en", output_lang="ru", return_indices=True)

print(f"Percentage of indices below 2100 bm25: {sum([1 for i in indices if i > 90000]) / len(indices) * 100:.2f}%")
