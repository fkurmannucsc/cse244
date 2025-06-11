import csv
import json
import os
import random
import re

import numpy
import torch


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_csv_file(path, delimiter=','):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        return list(reader)


def load_json_file(path):
    with open(path, "r") as file:
        return json.load(file)


def load_psl_file(path, dtype=str):
    data = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue

            data.append(list(map(dtype, line.split("\t"))))

    return data


def load_txt_file(path):
    with open(path, 'r') as file:
        return file.read()


def print_trainable_parameters(model: torch.nn.Module):
    trainable_params = 0
    all_param = 0
    # print(f"Model: {model.__class__.__name__}")
    # print(f"Model modules.")
    # for name, module in model.named_modules():
    #     print(f"Module: {name}")

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def write_csv_file(path, data, delimiter=','):
    with open(path, 'w') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(data)


def write_json_file(path, data, indent=4):
    with open(path, "w") as file:
        if indent is None:
            json.dump(data, file)
        else:
            json.dump(data, file, indent=indent)


def write_psl_file(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")


def save_model_state(model: torch.nn.Module, out_directory: str, filename):
    formatted_model_state_dict = {
        re.sub(r"^module\.", "", key).strip(): model.state_dict()[key]
        for key in model.state_dict()
    }
    torch.save(formatted_model_state_dict, os.path.join(out_directory, filename))


def seed_everything(seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
