"""
Handles interaction with the stored embeddings in
order to retrieve relevant data relating to model predictions.
"""

import csv
import datetime
import glob
import os
import random
import re
import sys
import torch

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from backend.constants import DB_DATA_DIR
from backend.constants import ANNOTATIONS_DIR
from backend.constants import FULL_ANNOTATIONS_PATH
from backend.constants import TRAIN_ANNOTATIONS_PATH
from backend.constants import VALID_ANNOTATIONS_PATH
from backend.constants import FULL_ANNOTATIONS
from backend.constants import TRAIN_ANNOTATIONS
from backend.constants import VALID_ANNOTATIONS
from backend.constants import RAW_DATA_DIR
from backend.constants import IMAGES_DIR
from backend.constants import EMBEDDINGS_DIR


import repo_utils
import backend.db.shark
import backend.utils

def check_raw_data(raw_annotations_file, verbose=False):
	"""
	Check raw-annotations file for issues and print stats.
	"""
	# Read in raw annotations file.
	raw_annotations_data = repo_utils.load_csv_file(os.path.join(RAW_DATA_DIR, "annotations", raw_annotations_file))
	raw_annotations_data = raw_annotations_data[1:]

	# Create a list containing raw_annotations_data items.
	parsed_raw_annotations = backend.utils.parse_raw_annotations(raw_annotations_data)

	num_entities = len(raw_annotations_data)
	num_annotations = len(parsed_raw_annotations)

	# Get all annotations
	annotations = []
	for key, value in parsed_raw_annotations.items():
		annotations.append(value["name"])
	
	# Detect duplicate annotations
	annotations_set = list(set(annotations))
	if (len(annotations_set) != len(annotations)):
		print("Duplicate annotations detected.")
		for item in annotations:
			if item not in annotations_set:
				print(item)

	# Compare set of images and set of annotations
	annotations_with_images = []
	annotations_without_images = []
	images_with_annotations = []
	images_without_annotations = []
	correctly_named = []
	incorrectly_named = []

	# Parse over annotations
	for finImage in annotations:
		if finImage + '.jpeg' not in os.listdir(IMAGES_DIR):
			annotations_without_images.append(finImage)
		else:
			annotations_with_images.append(finImage)
	
	# Parse over image files
	for finImage in os.listdir(IMAGES_DIR):
		finImage_name = finImage[:-5]
		if finImage_name not in annotations:
			images_without_annotations.append(finImage_name)
		else:
			images_with_annotations.append(finImage_name)

		if re.search(r"[A-Z]+\d\d\d\d\d\d\d\d$", finImage_name) is not None:
			correctly_named.append(finImage_name)
		else:
			incorrectly_named.append(finImage_name)
			
	# Report
	print("Total entities: ", num_entities)
	print("Total annotations", num_annotations)
	print("Total images: ", len(os.listdir(IMAGES_DIR)))

	print("Annotations with images: ", len(annotations_with_images))
	if (verbose == True):
		for item in annotations_with_images: 
			print(item)
	print("Annotations without images: ", len(annotations_without_images))
	if (verbose == True):
		for item in annotations_without_images: 
			print(item)
	print("Images with annotations: ", len(images_with_annotations))
	if (verbose == True):
		for item in images_with_annotations: 
			print(item)
	print("Images without annotations: ", len(images_without_annotations))
	if (verbose == True):
		for item in images_without_annotations: 
			print(item)
	print("Correctly named images: ", len(correctly_named))
	if (verbose == True):
		for item in correctly_named:
			print(item)
	print("Incorrectly named images: ", len(incorrectly_named))
	if (verbose == True):
		for item in incorrectly_named:
			print(item)

def format_image_directory(source_directory=RAW_DATA_DIR, verbose=False):
	"""
	Read in raw images and build image directory for GUI display and model use.
	"""
	os.makedirs(IMAGES_DIR, exist_ok=True)

	# Count images.
	total_images = 0
	for path in glob.iglob(os.path.join(source_directory, '**', '*'), recursive=True):
		filename = os.path.basename(path)
		if filename.split('.')[-1].lower().endswith('jpg') or filename.split('.')[-1].lower().endswith('png') or filename.split('.')[-1].lower().endswith('jpeg'):
			total_images += 1

	# Process all images, unifying format and suffix naming.
	image_index = 0
	for path in glob.iglob(os.path.join(source_directory, '**', '*'), recursive=True):
		filename = os.path.basename(path)

		if filename.split('.')[-1].lower().endswith('jpg') or filename.split('.')[-1].lower().endswith('png') or filename.split('.')[-1].lower().endswith('jpeg'):
			if (verbose == True):
				print("{} / {} Processing: {}".format(image_index, total_images, filename.ljust(50)), end='\r')
			
			out_filename = filename.split('.')[0].split('_')[0] + '.jpeg'
			if os.path.exists(os.path.join(IMAGES_DIR, out_filename)):
				continue

			try:
				image = Image.open(path).convert('RGB')
			except Exception as e:
				print(f"Error reading image {filename}: {e}")
				continue

			# Fixes -- libpng warning: iCCP: known incorrect sRGB profile
			image.info.pop('icc_profile', None)
			try:
				image.save(os.path.join(IMAGES_DIR, out_filename))
			except Exception as e:
				print(f"Error saving image {filename}: {e}")
			image.close()
			image_index += 1

	print(f"Loaded and converted {image_index} images.")


def prepare_dataset(save=True, train_test=True, full_clusters_removed=False, annotations_file=None):
	"""
	
	"""
	# Lists to contain needed data.
	ids = []
	filenames = []
	labels = []

	# Get the annotations file if given and limit the ids you are preparing to only those in that file.
	if annotations_file is not None:
		focus_ids = []
		focus_annotations = repo_utils.load_csv_file(annotations_file, delimiter='\t')
		for annotation in focus_annotations:
			focus_ids.append(int(annotation[0]))

	# Get all sharks and their associated medias.
	all_sharks = backend.db.shark.get_sharks(ids=None, get_full=True, get_source=True, remove_duplicates=True)
	shark_media_dict = {}
	for shark in all_sharks:
		media_dict = {"ids" : [], "filenames": [], "label": shark.id}
		for media in shark["associated_medias"]:
			media_dict["ids"].append(media.id)
			media_dict["filenames"].append(media.name + ".jpeg")

		# Check ids and filenames lists are same length.
		assert(len(media_dict["ids"]) == len(media_dict["filenames"]))

		shark_media_dict[shark.id] = media_dict

	# Create lists of image ids, labels, and filenames.
	for shark, media_dict in shark_media_dict.items():
		for index, id in enumerate(media_dict["ids"]):
			if annotations_file is None or id in focus_ids:
				ids.append(id)
				filenames.append(media_dict["filenames"][index])
				labels.append(shark)

	print(f"Number of ids in full dataset: {len(ids)}.")

	# Check ids and filenames lists are same length.
	assert(len(ids) == len(filenames) == len(labels))

	full_data = []
	train_data = []
	valid_data = []
	# Optionally, make the train/test split.
	if train_test:
		train_indices, test_indices = _partition_data(ids, labels, full_clusters_removed)
		print(f"Number of ids in train dataset: {len(train_indices)}, test dataset {len(test_indices)}.")

	for index, id in enumerate(ids):
		# Append all data to the full annotations file, then optionally split for train/valid.
		filename = filenames[index]
		label = labels[index]
		full_data.append([id, filename, label])
		
		if train_test:
			if index in train_indices:
				train_data.append([id, filename, label])
			elif index in test_indices:
				valid_data.append([id, filename, label])

	if save:
		# Save annotations.
		os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
		repo_utils.write_csv_file(FULL_ANNOTATIONS_PATH, full_data, delimiter='\t')
		if train_test:
			assert(len(train_data) + len(valid_data) == len(full_data)) # Check train and test add up to full.
		repo_utils.write_csv_file(TRAIN_ANNOTATIONS_PATH, train_data, delimiter='\t')
		repo_utils.write_csv_file(VALID_ANNOTATIONS_PATH, valid_data, delimiter='\t')
		
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		os.makedirs(os.path.join(ANNOTATIONS_DIR, timestamp), exist_ok=True)
		repo_utils.write_csv_file(os.path.join(ANNOTATIONS_DIR, timestamp, FULL_ANNOTATIONS), full_data, delimiter='\t')
		repo_utils.write_csv_file(os.path.join(ANNOTATIONS_DIR, timestamp, TRAIN_ANNOTATIONS), train_data, delimiter='\t')
		repo_utils.write_csv_file(os.path.join(ANNOTATIONS_DIR, timestamp, VALID_ANNOTATIONS), valid_data, delimiter='\t')
		
def _partition_data(ids, labels, full_clusters_removed=False):
	"""
	Split entire dataset into train and test sets with one validation image
	for each entity with greater than one associated image.
	If full_clusters_removed is called, you split instead by taking out a few
	complete clusters from the training set into the test set instead of the above split method.
	"""
	train_indices = []
	test_indices = []

	label_dict = {}

	for index, _ in enumerate(ids):
		if labels[index] not in label_dict.keys():
			label_dict[labels[index]] = []
		
		label_dict[labels[index]].append(index)

	for key, value in label_dict.items():
		# Single image sharks always to train.
		if len(value) == 1:
			train_indices.append(value[0])
		else:
			if full_clusters_removed:
				# Randomly set aside all images of a label to be in the validation set 
				random_int = random.randint(0, 9)
				if random_int <= 7:
					for index, item in enumerate(value):
						train_indices.append(item)
				else:
					for index, item in enumerate(value):
						test_indices.append(item)
			else:
				# Randomly select a single image for each label to be in the validation set.
				random_index = random.randint(0, len(value) - 1)
				for index, item in enumerate(value):
					if index != random_index:
						train_indices.append(item)
					else:
						test_indices.append(item)

	print(len(ids), len(labels), len(train_indices), len(test_indices))
	assert(len(ids) == len(labels) and (len(train_indices) + len(test_indices)))

	return train_indices, test_indices

def main():
	"""
	Main.
	"""
	# Returns only information.
	# check_raw_data(raw_annotations_file="database_test_annotations.csv")
	# check_raw_data(raw_annotations_file="finMatch_combined_2006-2024-v7.csv")
	# check_raw_data(raw_annotations_file="raw_annotations-updated.csv")
	# check_raw_data(raw_annotations_file="finMatch_combined_2006-2024-v7.csv")
	
	# Generate formatted annotations, split the data into a train and test set.
	prepare_dataset(save=True, train_test=False, full_clusters_removed=False)

	
	# # Populate image directory.
	# format_image_directory()

if __name__ == "__main__":
	main()
