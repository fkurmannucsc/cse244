"""
Utility functions for backend code.
"""

import datetime
import glob
import hashlib
import json
import logging
import os
import random
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
import torch

from argon2 import PasswordHasher
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from backend.config import BACKBONE_CONSTRUCTORS
from backend.config import LORA_CONFIG
from backend.config import LORA_CONFIG_CONSTRUCTOR
from backend.config import PROJECTION_HEAD_CONSTRUCTORS
from backend.constants import BASE_MODELS_DIR
from backend.constants import DB_PATH
from backend.constants import SUCCESS
from backend.constants import ERROR
from backend.constants import BASE_DATA_DIR
from backend.constants import BASE_MODELS_DIR
from backend.constants import DB_PATH

from backend.sharkfin_dataset import SharkfinDataset # Import the dataset class
import backend.db.media
import repo_utils

# TODO, change this from clipper copy
SHARK_TAG_NAME = 'shark-clipper'
TABLE_NAMES = ["Sharks", "Media", "Tags", "AlternateNames", "Labels", "Users", "Tokens", "Actions"]

# Utils functions.
def load_model(model_name):
    """
    Load a pretrained model and return the projection head and backbone for use.
    """
    device = repo_utils.get_torch_device()

    # Load the pretrained model.
    print(f"Loading model: {model_name}")
    model_config = repo_utils.load_json_file(os.path.join(BASE_MODELS_DIR, "trained", model_name, "config.json"))
    backbone_config = model_config["model"]["backbone"]
    projection_head_config = model_config["model"]["projection_head"]

    # Instantiate the base backbone
    # The ViTBackbone constructor handles 4-channel input modifications.
    base_vit_model = BACKBONE_CONSTRUCTORS[backbone_config["name"]](**backbone_config["options"]).to(device)

    lora_adapter_path = os.path.join(BASE_MODELS_DIR, "trained", model_name, "best-backbone-lora")
    frozen_state_dict_path = os.path.join(BASE_MODELS_DIR, "trained", model_name, "best-backbone-state.pt")

    model_backbone = None
    if os.path.isdir(lora_adapter_path):
        print(f"Found LoRA adapters at {lora_adapter_path}. Loading PeftModel.")
        model_backbone = PeftModel.from_pretrained(base_vit_model, lora_adapter_path).to(device)
    elif os.path.isfile(frozen_state_dict_path):
        print(f"Found frozen backbone state_dict at {frozen_state_dict_path}. Loading state_dict.")
        base_vit_model.load_state_dict(torch.load(frozen_state_dict_path, map_location=device))
        model_backbone = base_vit_model # The base model itself is the backbone
    else:
        # Fallback or error if neither LoRA nor frozen state is found.
        # This might indicate an issue with the saved model or model_name.
        print(f"Warning: Neither LoRA adapters nor frozen state_dict found for backbone in {model_name}. Using base initialized model.")
        # Depending on requirements, you might want to raise an error here.
        # For now, it will proceed with the randomly initialized base_vit_model if no weights are found.
        model_backbone = base_vit_model

    model_backbone.eval()
    projection_head = PROJECTION_HEAD_CONSTRUCTORS[projection_head_config["name"]](input_dim=model_backbone.get_output_dim(), **projection_head_config["options"]).to(device)
    projection_head.load_state_dict(torch.load(os.path.join(BASE_MODELS_DIR, "trained", model_name, "best-projection-head.pt"), map_location=device, weights_only=False))
    projection_head.eval()

    return (projection_head, model_backbone)

def execute_sql(sql, data=None, commit=False):
    """
    Helper function to execute sql commands. TODO make the commit optional.
    """
    # try:
    connection = db_connect()

    # TODO Correct foreign key enforcement.
    # connection.execute("PRAGMA foreign_keys = 1")

    with connection:
        if data is None:
            result = connection.execute(sql)
        else:
            result = connection.execute(sql, data)
        result = result.fetchall()

    connection.commit()
    connection.close()

    return result

    # except:
    #     return ERROR

def build_sql_placeholders(values_list, basename="x") -> tuple[str, dict]:
    """
    Helper function to prepare named placedholers and value dict for sql queries.
    """
    placeholders = ", ".join([f":{basename}_{i}" for i, x in enumerate(values_list)])
    values = {f"{basename}_{i}": x for i, x in enumerate(values_list)}
    return f"({placeholders})", values

def list_to_string(list, delimiter=", "):
    return_string = ""
    for item in list:
        return_string = return_string + str(item) + delimiter

    return_string = return_string[:-len(delimiter)]
    return return_string

def clean_input_list(input_list, remove_item=""):
    if not isinstance(input_list, list):
        return None
    # Remove all instances of remove_item
    while remove_item in input_list:
         input_list.remove(remove_item)

    # Remove leading spaces (to allow commas followed by spaces in search).
    for index, item in enumerate(input_list):
        while isinstance(item, str) and item[0] == " ":
            item = item[1:]
        input_list[index] = item

    return input_list

def db_connect():
    connection = sqlite3.connect(DB_PATH)
    return (connection)

def get_all_ids(table):
    if table not in TABLE_NAMES:
        raise ValueError("Error, invalid database table.")

    sql = f"""SELECT id FROM {table}"""

    result = execute_sql(sql)
    if result is None:
        return ERROR

    return [item[0] for item in result]

def id_exists(id, table):
    """ 
    Return true if an item with the specified ID exists in a table, else false.
    """
    if table not in TABLE_NAMES:
        raise ValueError("Error, invalid database table.")

    data = ({"id": id})
    sql = f"""SELECT COUNT(*) FROM {table} WHERE id = :id;"""

    result = execute_sql(sql, data)
    if result is None:
        return False

    count = result[0]
    return False if count == 0 else True

# Parsing functions
def parse_raw_annotations(raw_annotations_data):
    """Return a dictionary with image names as keys and their associated metadata as values."""
    return_dict = {}
    for index, item in enumerate(raw_annotations_data):
        sex = item[1]
        length = item[2]
        tags = item[3]
        nicknames = item[4]
        quality = item[5]

        # Metadata for reference media.
        for media in [item[0]]:
            if media != "":
                # Get all metadata from name
                media_metadata_dict = parse_media_name(media)
                finImage_metadata_dict = {
                    "shark": index,
                    "name": media,
                    "length": length,
                    "sex": sex,
                    "time": media_metadata_dict["time"],
                    "location": media_metadata_dict["location"],
                    "quality": quality,
                    "tags": tags,
                    "nicknames": nicknames
                }
                return_dict[media] = finImage_metadata_dict
        
        # Metadata for remaining media.
        for media in item[6:]:
            if media != "":
                # Get all metadata from name
                media_metadata_dict = parse_media_name(media)
                finImage_metadata_dict = {
                    "shark": index,
                    "name": media,
                    "length": length,
                    "sex": sex,
                    "time": media_metadata_dict["time"],
                    "location": media_metadata_dict["location"],
                    "quality": quality,
                    "tags": "",
                    "nicknames": ""
                }
                return_dict[media] = finImage_metadata_dict

    return return_dict

def parse_media_name(media_name):
	"""Parse media name for all possible data."""
	metadata_dict = {
		"name": media_name,
		"media_name": None,
		"location": None,
		"time": None,
		"length": None,
		"sex": None,
		"extension": None,
	}
	
	# If name is correctly formatted, extract information.
	match = re.search(r"^([A-Z][A-Z]+)(\d\d\d\d\d\d\d\d)_*([A-Za-a])*_*(\d[.\d]*)*\.*([A-Za-z]+)*$", media_name)
	if (match is not None):
			metadata_dict["media_name"] = match.group(1) + match.group(2)
			metadata_dict["location"] = match.group(1)
			numerical = match.group(2)
			metadata_dict["sex"] = match.group(3)
			metadata_dict["length"] = match.group(4)
			metadata_dict["extension"] = match.group(5)
	else:
		return metadata_dict
	
	# Parse numerical information.
	year = numerical[0:2]
	if (int(year) < 50):
		year = '20' + year
	else:
		year = '19' + year
	month = numerical[2:4]
	day = numerical[4:6]
	date = year + '-' + month + '-' + day
	try:
		metadata_dict["time"] = datetime.datetime.strptime(date, '%Y-%m-%d')
	except:
		print("ERROR: Unable to parse name for date:", media_name)
		metadata_dict["time"] = None
	
	return metadata_dict

# Get metadata for all images
def parse_media_medatada(media_directory):
    """Get metadata for all images."""
    for path in glob.iglob(os.path.join(media_directory, '**', '*'), recursive=True):
        filename = os.path.basename(path)
        if os.path.isfile(path):
            print(path)

            image_medatada = get_all_metadata(path)
            print(filename, image_medatada)

# File information parsing: TODO

# Code adapted from Shark Clipper, Eriq Augustine
# Get the metadata from a file.
# The metadata will be in a dict, but there are no guarentees on the structure,
# it will just be what ffprobe returns.
def get_all_metadata(path):
    args = [
        _get_path('ffprobe'),
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        path,
    ]

    stdout, _ = _run(args, 'ffprobe')
    return json.loads(stdout)

# Get metadata that we believe is important.
def get_key_metadata(path, override_with_shark_data = True):
    data = {}
    all_metadata = get_all_metadata(path)

    if ('format' in all_metadata):
        tags = all_metadata['format'].get('tags', {})

        if ('location' in tags):
            text_location = tags['location'].strip()
            data['location_raw'] = text_location

            match = re.search(r'^((?:\+|-)?\d+\.\d+)((?:\+|-)?\d+\.\d+)/?$', text_location)
            if (match is not None):
                data['location'] = {
                    'latitude': match.group(1),
                    'longitude': match.group(2),
                }

        if ('creation_time' in tags):
            text_time = tags['creation_time'].strip()
            data['start_time_raw'] = text_time

            try:
                parsed_time = datetime.datetime.fromisoformat(text_time)
                data['start_time_unix'] = int(parsed_time.timestamp())
            except Exception as ex:
                logging.warn("Invalid start/creation time found in file metadata.", exc_info = ex)

        if ((override_with_shark_data) and (SHARK_TAG_NAME in tags)):
            shark_data_str = tags[SHARK_TAG_NAME].strip()

            try:
                shark_data = json.loads(shark_data_str)
            except Exception as ex:
                logging.warn("Invalid (json) shark data found in file metadata.", exc_info = ex)

            data.update(shark_data)

    return data, all_metadata

def _run(args, name):
	logging.debug(shlex.join(args))

	result = subprocess.run(args, capture_output = True)
	if (result.returncode != 0):
		raise ValueError("%s did not exit cleanly.\n--- stdout ---\n%s\n---\n--- stderr ---%s\n---" % (name, result.stdout, result.stderr))

	return result.stdout, result.stderr

def _get_path(name):
	# First, do a standard path search.
	print(name)
	path = shutil.which(name)
	if (path is not None):
		return path

	# # Next, check the local (project) path.
	path = os.path.join(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(os.path.realpath(__file__))), '..', '..')), name)
	if (os.path.isfile(path)):
		return path

	return None

def handle_error(error):
    print(error)

# Debugging.
def print_password(cleartext, salt):
    hash = hashlib.sha256(str.encode(cleartext)).hexdigest()
    ph = PasswordHasher()
    crypto_hash = ph.hash(hash, salt=salt.encode("utf-8"))
    print(f"clear: {cleartext}, hash: {hash}, crypto: {crypto_hash}")

# Data processing.
def rename_files_with_prefix(root_dir):
    """
    Finds all files with a specific prefix in a nested directory structure
    and renames them with a new prefix.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            match_object = re.search(r"[A-Z][A-Z]\d\d1[3|4]", filename)
            if match_object is not None:
                old_prefix = match_object.group(0)
                year = int(old_prefix[2:4])
                month = int(old_prefix[4:6])

                month = month % 12
                year = year + 1

                month_string = str(month)
                year_string = str(year)

                if len(month_string) == 1:
                     month_string = "0" + month_string
                if len(year_string) == 1:
                     year_string = "0" + year_string

                new_prefix = old_prefix[0:2] + year_string + month_string
                # print(old_prefix, new_prefix)
                old_filepath = os.path.join(dirpath, filename)
                new_filename = new_prefix + filename[6:] #slice off the old prefix

                print(filename, new_filename)
                new_filepath = os.path.join(dirpath, new_filename)

                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"Renamed: {old_filepath} -> {new_filepath}")
                except OSError as e:
                    print(f"Error renaming {old_filepath}: {e}")

# Data hacking.
def find_and_replace(root_dir):
    """
    Find and replace all instances of a regular expression in a file.
    """
    lines = repo_utils.load_csv_file("data-out/database-format/media.csv")
    new_lines = []

    for list in lines:
        new_list = []
        for line in list:
            match_object = re.search(r"[A-Z][A-Z]\d\d1[3|4]", line)
            if match_object is not None:
                old_prefix = match_object.group(0)
                year = int(old_prefix[2:4])
                month = int(old_prefix[4:6])

                month = month % 12
                year = year + 1

                month_string = str(month)
                year_string = str(year)

                if len(month_string) == 1:
                        month_string = "0" + month_string
                if len(year_string) == 1:
                        year_string = "0" + year_string

                new_prefix = old_prefix[0:2] + year_string + month_string
                print(old_prefix, new_prefix)

                new_line = line.replace(old_prefix, new_prefix)
                new_list.append(new_line)

            else:
                new_list.append(line)
        new_lines.append(new_list)

    repo_utils.write_csv_file("data-out/database-format/media-new.csv", new_lines)

def changed_names():
    input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../renamed.txt")
    input_file2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../renamed2.txt")
    corrected = []
    with open (input_file2) as file:
        for line in file:
            corrected.append(line.strip())
    then_now = []
    with open (input_file) as file:
        for line in file:
            line = line.strip()
            for item in corrected:
                if item in line:
                    line = line.split()
                    clean_line = []
                    for item in line:
                        item = item.split(".")
                        clean_line.append(item[0])
                    then_now.append(clean_line)

    actions = []
    number = 436
    for item in then_now:
        id = backend.db.media.get_media_id_by_name(item[1])
        media = backend.db.media.get_media(id)
        output = media.to_dict()
        media.name = item[0]
        media.filename = item[0] + ".jpeg"
        source = media.to_dict()
        action_notes = json.dumps({"original": source, "final": output})
        
        actions.append([number, id, id, "update", "media", "sammy", "1740830400", action_notes])
        number += 1
    for item in actions:
        print(item)
    
    repo_utils.write_csv_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), "name_updates.csv"), actions, delimiter=",")

def add_times():
    lines = repo_utils.load_csv_file("data-out/database-format/media.csv")
    new_lines = []
    for item in lines:
        if item[-3] == "":
            print(item)
            media_dict = parse_media_name(item[2])
            if (media_dict["time"] is not None):
                datetime = int((time.mktime(media_dict["time"].timetuple())))
                print(datetime)
                item[-3] = datetime
                new_lines.append(item)
            else:
                new_lines.append(item)
        else:
            new_lines.append(item)

    print(len(lines))
    print(len(new_lines))
    repo_utils.write_csv_file("data-out/database-format/media-new.csv", new_lines)

def add_types():
    lines = repo_utils.load_csv_file("data-out/database-format/tags.csv")
    new_lines = []
    for item in lines:
        item.append("")
        new_lines.append(item)
        # if item[-5] == "":
        #     item[-5] = "U"
        #     new_lines.append(item)
        # else:
        #     print(item)
        #     new_lines.append(item)

    print(len(lines))
    print(len(new_lines))
    repo_utils.write_csv_file("data-out/database-format/tags-new.csv", new_lines)

def main():
    add_types()
    # changed_names()
    # rename_files_with_prefix("data/raw-data/images")
    # find_and_replace(None)
    # print_password("shark", "c3063b8354d0c178dd4ab7e163a35bd0d4d82876ec4ec03fc8caf28116d04e82")
    # print_password("shark", "442d32839de05b00f53853a331d756abb0ca3bb080521aaeaa763a1e66d74f56")
    # print_password("shark", "4382daadb6a674ff798bd24762acf919afd77f593609d26bc6f7fdeb7cfa5b75")
    # print_password("shark", "92b146c222c1e5830e12bdacd629a5778647f0fc65b424b4d8e656d12e7db04e")
    # print_password("shark", "26ae729585d28b59c8cba2c0e3cc71fe48a79ccb7097b085eb45e49da34cc9d5")
    # print_password("test", "676b3c67f55d20ed9a49e481d0f435470a39621c81d1d709652a25ff9e577fa6")
if __name__ == "__main__":
	main()
