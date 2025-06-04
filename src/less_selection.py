import argparse
from copy import deepcopy
import json
import os
from hashlib import md5
import re
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.data import DataCollatorForSeq2Seq
from torch.utils.data  import DataLoader
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import matplotlib.pyplot as plt

def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (either representation or gradients) has been stored. 

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                   attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model, batch):
    """ obtain gradients. """
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, batch):
    """ obtain gradients with sign. """
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None):
    """
    Collects gradients from the model during edevuation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for edevuation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 8  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interdev = 8  # project every 8 batches
    save_interdev = 160  # save every 160 batches

    def _project(current_full_grads, projected_grads):
        # print(current_full_grads[0].shape)
        current_full_grads = torch.stack(current_full_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(
                current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(
                f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)

    # never made it work sadly
    # fmodel, params, buffers = make_functional_with_buffers(model)
    # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)

    # initialize a project for each target projector dimension
    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size)
        projectors.append(proj)

    count = 0

    # set up a output directory for each dimension
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    # max index for each dimension
    max_index = min(get_max_saved_index(
        output_dirs[dim], "grads") for dim in proj_dim)

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

    for batch in tqdm(dataloader, total=len(dataloader)):
        prepare_batch(batch)
        count += 1

        if count <= max_index:
            if count%100==0:
                print("skipping count", count)
            continue

        try:
            if gradient_type == "adam":
                if count == 1:
                    print("Using Adam gradients")
                vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
            elif gradient_type == "sign":
                if count == 1:
                    print("Using Sign gradients")
                vectorized_grads = obtain_sign_gradients(model, batch)
            else:
                if count == 1:
                    print("Using SGD gradients")
                vectorized_grads = obtain_gradients(model, batch)

            # add the gradients to the full_grads
            full_grads.append(vectorized_grads)
            model.zero_grad()

            if count % project_interdev == 0:
                _project(full_grads, projected_grads)
                full_grads = []

            if count % save_interdev == 0:
                _save(projected_grads, output_dirs)
        except Exception as e:
            print(f"Error at row {count-1}: {e}")
            exit(1)

        if max_samples is not None and count == max_samples:
            break

    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    print(projected_grads.keys())

    for dim in proj_dim:
        _save(projected_grads, output_dirs)

    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

    print("Finished")


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def get_loss(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             output_dir: str,):
    """ Get the loss of the model on the given dataset. """
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        prepare_batch(batch)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (
        total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))

def main():
    """
    Main function to collect the gradient and representation, and select accordng to the representation
    """
    parser = argparse.ArgumentParser(description='Gradient/Representation Collection')

    # Model
    parser.add_argument('--model_name', required=True, type=str,
                        help='Pretrained model name')
    parser.add_argument('--reversed_model_name', type=str,
                        help='Reversed model name')
    # Arguments for training dataset, including:
    # Is training dataset from huggingface or local
    # Path to the training dataset / Name of the training dataset from huggingface
    parser.add_argument('--train_dataset', required=True, type=str,
                        help='Path to the training dataset')
    parser.add_argument('--train_from_disk', action='store_true',
                        help='If the training dataset is local')
    parser.add_argument('--train_split', type=str, default="train",
                        help='The split of the training dataset to use')
    # Arguments for devdataset, including:
    # Is dev dataset from huggingface or local
    # Path to the dev dataset / Name of the dev dataset from huggingface
    parser.add_argument('--dev_dataset', required=True, type=str,
                        help='Path to the dev dataset')
    parser.add_argument('--dev_from_disk', action='store_true',
                        help='If the dev dataset is local')
    parser.add_argument('--dev_split', type=str, default="dev",
                        help='The split of the dev dataset to use')
    # Output directory
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Where to save the grads etc')
    parser.add_argument('--dataset_output_dir', required=True, type=str,
                        help='Where to save the selected dataset')

    parser.add_argument('--ignore_first_2100', action='store_true')

    args = parser.parse_args()

    if args.train_dataset == "../data/mixed_dataset":
        flawed_rows = [4182, 7679, 7687, 10930, 10972, 10978, 11025, 15293, 16796, 16810, 19303, 19307, 21029, 21196, 21245, 21358, 21390, 28291, 36371, 39482, 65412, 66223, 73638, 76373, 84776, 87519]
        print("Flawed rows: ", flawed_rows)

    # Check if CUDA is available and not disabled
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected! Training will be slow.")
        raise RuntimeError("No GPU detected! Training will be slow.")

    # load the model, from disk or from huggingface
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, torch_dtype=torch.float)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    reversed_tokenizer = AutoTokenizer.from_pretrained(
        args.reversed_model_name) if args.reversed_model_name else tokenizer
    model = model.to(device)
    
    if args.train_from_disk:
        train_dataset = load_from_disk(args.train_dataset)
        train_dataset = train_dataset[args.train_split]
    else:
        train_dataset = load_dataset(args.train_dataset, split=args.train_split)
    
    # replace flawed rows with dummy sentences
    def clear_row(example, idx):
        if idx in flawed_rows:
            example["en"] = ""
            example["ru"] = ""
        return example
    train_dataset = train_dataset.map(clear_row, with_indices=True)

    # Backup
    original_train_dataset = deepcopy(train_dataset)

    if args.dev_from_disk:
        dev_dataset = load_from_disk(args.dev_dataset)
        dev_dataset = dev_dataset[args.dev_split]
    else:
        dev_dataset = load_dataset(args.dev_dataset, split=args.dev_split)

    # tokenize the dataset into input_ids, attention_mask, and labels
    # this is a translation task
    # input_ids are from the coulmn "en"
    # labels are from the column "ru"
    # returned input_ids and attention_mask should be tensors
    def tokenize_function(examples):
        pattern = r"[^A-Za-z\u00C0-\u024F\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\s.,!?:;\"'()\[\]\{\}\-]"

        def clean_text(text):
            if isinstance(text, list):
                return [re.sub(pattern, '', t) for t in text]
            else:
                return re.sub(pattern, '', text)
        inputs = clean_text(examples["en"])
        targets = clean_text(examples["ru"])
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels_tokenized = reversed_tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels_tokenized["input_ids"]
        # Convert lists to tensors
        # for k in model_inputs:
        #     model_inputs[k] = torch.tensor(model_inputs[k])
        return model_inputs
    # tokenize, but don't keep the columns "en" and "ru"
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.remove_columns(["en", "ru"])
    dev_dataset = dev_dataset.map(tokenize_function, batched=True)
    dev_dataset = dev_dataset.remove_columns(["en", "ru"])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors="pt")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

    train_output_dir = os.path.join(args.output_dir, "train")
    dev_output_dir = os.path.join(args.output_dir, "dev")

    collect_grads(
        train_loader,
        model,
        train_output_dir,
        proj_dim=[8192],
        gradient_type="sgd",
        # max_samples=1000
    )
    collect_grads(
        dev_loader,
        model,
        dev_output_dir,
        proj_dim=[8192],
        gradient_type="sgd",
        # max_samples=1000
    )

    print("Finished collecting gradients")

    def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
        """Calculate the influence score.

        Args:
            training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
            validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
        """
        # N x N_VALID
        influence_scores = torch.matmul(
            training_info, validation_info.transpose(0, 1))
        return influence_scores


    # calculate the influence score for each validation task
    validation_path = os.path.join(dev_output_dir, "dim8192")
    if os.path.isdir(validation_path):
        validation_path = os.path.join(validation_path, "all_orig.pt")
    validation_info = torch.load(validation_path)
    dev_percentage = 0.1
    # shuffle and select
    torch.manual_seed(42)
    idx = torch.randperm(validation_info.shape[0])
    selected_idx = idx[:int(validation_info.shape[0] * dev_percentage)]
    validation_info = validation_info[selected_idx]
    # normalize on the second dimension by L2
    validation_info = F.normalize(validation_info, dim=1)

    if not torch.is_tensor(validation_info):
        validation_info = torch.tensor(validation_info)
    validation_info = validation_info.to(device).float()
    gradient_path = os.path.join(train_output_dir, "dim8192")
    if os.path.isdir(gradient_path):
        gradient_path = os.path.join(gradient_path, "all_orig.pt")
    training_info = torch.load(gradient_path)
    training_info = F.normalize(training_info, dim=1)

    training_info = training_info.to(device).float()

    influence_score = calculate_influence_score(
            training_info=training_info, validation_info=validation_info)
    print(influence_score.shape)
    influence_score = influence_score.max(-1)[0]
    score_dir = os.path.join(args.output_dir, "influence_score")
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    output_file = os.path.join(score_dir, f"influence_score.pt")
    torch.save(influence_score, output_file)
    print("Saved influence score to {}".format(output_file))

    # select the top k samples
    train_percentage = 0.2
    train_size = int(training_info.shape[0] * train_percentage)
    print(train_size)

    # histogram of influence_score
    plt.hist(influence_score.cpu().numpy(), bins=100, label="all", color="blue")
    # also hist idx<2100 and idx>=2100 seperately
    plt.hist(influence_score[:2100].cpu().numpy(), bins=100, alpha=0.5, label="idx<2100", color="red")
    plt.hist(influence_score[2100:].cpu().numpy(), bins=100, alpha=0.5, label="idx>=2100", color="green")
    plt.xlabel("Influence Score")
    plt.ylabel("Count")
    plt.title("Influence Score Histogram")
    plt.legend()
    plt.savefig(os.path.join(score_dir, "influence_score_histogram.png"))

    if args.ignore_first_2100:
        # ignore the first 2100 samples
        print("Ignoring the first 2100 samples")
        influence_score[:2100] = -100
    selected_idx = torch.topk(influence_score, train_size)[1]
    # print(original_train_dataset)
    print(selected_idx.tolist())
    # see how many idx are <2100
    print("Number of selected idx < 2100: ", sum([
        1 for idx in selected_idx if idx < 2100]))
    selected_dataset = DatasetDict()
    selected_dataset["train"] = original_train_dataset.select(selected_idx.tolist())
    # print the first few lines
    print(selected_dataset["train"][:20])
    # save the selected dataset
    selected_dataset.save_to_disk(args.dataset_output_dir)


if __name__ == "__main__":
    main()

# python less_selection.py --model_name=facebook/wmt19-en-ru --train_dataset=sethjsa/scipar_en_ru_parallel --dev_dataset=sethjsa/medline_ru_parallel --dev_split=train --output_dir=../grads/example