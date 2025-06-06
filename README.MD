# Robust Domain Adaptation for Low-Resource Biomedical Translation

Welcome to the official repository for our reproducibility research of "**Robust Domain Adaptation for Low-Resource Biomedical Translation**".

Download the repository as a ```.zip``` or clone the repository.
Install the correct version of the used packages from the .yml file using the following command:
```console
foo@bar:~$ conda env create -f environment.yml
```
Upon installation of the environment, it can be (de)activated using:
```console
foo@bar:~$ conda activate nlp2
(nlp2) foo@bar:~$ conda deactivate nlp2
```
The environment can be deleted using:
```console
foo@bar:~$ conda remove -n nlp2 --all
```
Additional packages can be installed using pip:
```console
foo@bar:~$ pip install <package>
```

### Datasets
Many differerent datasets are used, these are taken directly from huggingface, and are loaded automatically. For constructing a combined or partially corrupted dataset, the following python files can be used:
```
(nlp2) foo@bar:~$ python create_corrupted_mixed_medline_scipar.py
(nlp2) foo@bar:~$ python create_corrupted_mixed_scipar_tico.py
(nlp2) foo@bar:~$ python create_mixed_medline_scipar_dataset.py
(nlp2) foo@bar:~$ python create_mixed_scipar_tico_dataset.py
(nlp2) foo@bar:~$ python create_mixed_wikipedia_scipar_dataset_small.py
(nlp2) foo@bar:~$ python create_mixed_wikipedia_scipar_dataset.py
```

### Visualizations

For all visualizations, the notebook *data_selection_methods.ipynb* was used. 

## How to run

For each of the experiments we will provide the basic scripts needed to run the code. The scripts we used to run experiments cna be found in the experiments folder. These scripts can easily be turned into shell scripts to allow execution on other machines (don't forget to change all paths according to your filenames and filesystems)

### Training a model
```bash
cd ../src

python -m train --train_dataset=sethjsa/medline_ru_parallel --output_dir=../results/wmt_ft --warmup_steps=100 --model_name=facebook/wmt19-en-ru --reversed_model_name=facebook/wmt19-ru-en --num_train_epochs=5
```

### Evaluating a model
```bash
cd ../src

python -m evaluate_model --model_name=facebook/wmt19-en-ru --reversed_model_name=facebook/wmt19-ru-en --test_dataset=sethjsa/wmt20bio_en_ru_sent
```
### Backtranslating a dataset
Below we show how backtranslation is performed with a specific temperature and augmentation level, in this case a temperature of 2.0 and an augmentation level of 0.05, note the definition of an output directory, this must be changed for every model.  
```bash
cd ../src

python -m backtranslate --model_name=facebook/wmt19-ru-en --output_dir=../data/bt_medline_temp_2.0_augmented_0.05/train --temperature 2.0 --augment_data 0.05
python -m train --train_dataset=../data/bt_medline_temp_2.0_augmented_0.05 --train_from_disk --output_dir=../results/wmt_bt_temp_2.0_augmented_0.05 --warmup_steps=100 --model_name=facebook/wmt19-en-ru --reversed_model_name=facebook/wmt19-ru-en --num_train_epochs=5
python -m evaluate_model --model_name=../results/wmt_bt_temp_2.0_augmented_0.05 --reversed_model_name=facebook/wmt19-ru-en --test_dataset=sethjsa/wmt20bio_en_ru_sent
```

# Acknowledgements
The code in this repository is partially based on the implementation of backtranslation and fine-tuning by Seth Aycock as part of the project course NLP2. 
