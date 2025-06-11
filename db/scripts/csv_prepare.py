
import hashlib
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from backend.constants import DB_DATA_DIR
from backend.constants import FULL_ANNOTATIONS_PATH
from backend.constants import RAW_DATA_DIR
from backend.constants import IMAGES_DIR

from repo_utils import load_csv_file
from repo_utils import write_csv_file
from backend.utils import parse_raw_annotations

def prepare_table_csvs(raw_annotations_file, annotations_file, media_directory):
	"""
	Read in data from annotations files and write information to CSV files
	that correspond to database tables to allow use of .import to populate database.
	"""
	# Read in raw annotations file.
	raw_annotations_data = load_csv_file(os.path.join(RAW_DATA_DIR, "annotations", raw_annotations_file))
	raw_annotations_data = raw_annotations_data[1:]

	# Create a list containing raw_annotations_data items.
	parsed_raw_annotations = parse_raw_annotations(raw_annotations_data)

	# Read in formatted annotations file.
	annotations_data = load_csv_file(annotations_file, delimiter="\t")

	# Create table formatted csv files, no header columns.
	os.makedirs(DB_DATA_DIR, exist_ok=True)
		
	media_list = prepare_media(parsed_raw_annotations, annotations_data)
	write_csv_file(os.path.join(DB_DATA_DIR, "media.csv"), media_list, delimiter=",")

	sharks_list, media_shark_reference = prepare_sharks(parsed_raw_annotations, annotations_data)
	write_csv_file(os.path.join(DB_DATA_DIR, "sharks.csv"), sharks_list, delimiter=",")

	alternateNames_list = prepare_alternateNames(parsed_raw_annotations, annotations_data, media_shark_reference)
	write_csv_file(os.path.join(DB_DATA_DIR, "alternateNames.csv"), alternateNames_list, delimiter=",")

	tags_list = prepare_tags(parsed_raw_annotations, annotations_data, media_shark_reference)
	write_csv_file(os.path.join(DB_DATA_DIR, "tags.csv"), tags_list, delimiter=",")
	
	users_list = [
		[0, "Fabrice Kurmann", "fkurmann@ucsc.edu", "fabrice", "owner", "$argon2id$v=19$m=65536,t=3,p=4$Y2MzMmRhN2NmODc3MDA2ODM5NzY4ZDE5NzhkYTQzYzI5MTk1NGI4NjcxZTU3ZGI3NzQwYjQ3MGFmYmU0NGVhMA$1g3DR5tzs0wEb1mqps0eTjCdlPpZjl2pvt8wz0+2CjY", "cc32da7cf877006839768d1978da43c291954b8671e57db7740b470afbe44ea0"],
		[1, "Eriq Augustine", "eaugusti@ucsc.edu", "eriq", "dev", "$argon2id$v=19$m=65536,t=3,p=4$YmJiYThjMWY5MjgyYjFlNzQzZGI5YmNlNWExN2I1NjQ0MWVmODU3OTYzY2M1MmExNGRlNWUwNzQ3MmJmYTUxMQ$hpWjPlP4UGF6LUs1n7Kl1Ri9ev/WdhK7i8G8hOeGSPo", "bbba8c1f9282b1e743db9bce5a17b56441ef857963cc52a14de5e07472bfa511"],
		[2, "Samantha Andrzejaczek", "sammyaz@stanford.edu", "sammy", "admin", "$argon2id$v=19$m=65536,t=3,p=4$OWYxM2M4ZWIyNDM0NzRlZDc5MWVmYWVlOWJlNTNlZTNkNTkwNTdkZTYwNjg2OWM0ZTMyYThlNjQyYzg0NTliZA$PWvyizgDpHsR0e5vC5Q1W7VXXaAA9kyRl5UITUfSliU", "9f13c8eb243474ed791efaee9be53ee3d59057de606869c4e32a8e642c8459bd"],
		[3, "Alexandra DiGiacomo", "alexandra.digiacomo@stanford.edu", "alexandra", "admin", "$argon2id$v=19$m=65536,t=3,p=4$YzMwNjNiODM1NGQwYzE3OGRkNGFiN2UxNjNhMzViZDBkNGQ4Mjg3NmVjNGVjMDNmYzhjYWYyODExNmQwNGU4Mg$+Ctjt8kIYwREc91L7d0UGW8PdegS3m6ExcoKLB9TVR0", "c3063b8354d0c178dd4ab7e163a35bd0d4d82876ec4ec03fc8caf28116d04e82"],
		[4, "Barbara Block", "bblock@stanford.edu", "barbara", "admin", "$argon2id$v=19$m=65536,t=3,p=4$NDQyZDMyODM5ZGUwNWIwMGY1Mzg1M2EzMzFkNzU2YWJiMGNhM2JiMDgwNTIxYWFlYWE3NjNhMWU2NmQ3NGY1Ng$ibz+J5t1mruSB0VsRty1xwuwZDaqnZGMPE98b1bYjes", "442d32839de05b00f53853a331d756abb0ca3bb080521aaeaa763a1e66d74f56"],
		[5, "Lise Getoor", "getoor@ucsc.edu", "lise", "admin", "$argon2id$v=19$m=65536,t=3,p=4$NDM4MmRhYWRiNmE2NzRmZjc5OGJkMjQ3NjJhY2Y5MTlhZmQ3N2Y1OTM2MDlkMjZiYzZmN2ZkZWI3Y2ZhNWI3NQ$J43rLnGR+uo0xOQhUrxlhwUAkehjnzh869ZL41wZjfo", "4382daadb6a674ff798bd24762acf919afd77f593609d26bc6f7fdeb7cfa5b75"],
		[6, "Jesse Rodriguez", "jesser18@stanford.edu", "jesse", "admin", "$argon2id$v=19$m=65536,t=3,p=4$OTJiMTQ2YzIyMmMxZTU4MzBlMTJiZGFjZDYyOWE1Nzc4NjQ3ZjBmYzY1YjQyNGI0ZDhlNjU2ZDEyZTdkYjA0ZQ$ep6ivF1XKWvXW6H2pOowS+3pUAOOfkMled1BakSvroE", "92b146c222c1e5830e12bdacd629a5778647f0fc65b424b4d8e656d12e7db04e"],
		[7, "Connor Pryor", "cfpryor@ucsc.edu", "connor", "admin", "$argon2id$v=19$m=65536,t=3,p=4$MjZhZTcyOTU4NWQyOGI1OWM4Y2JhMmMwZTNjYzcxZmU0OGE3OWNjYjcwOTdiMDg1ZWI0NWU0OWRhMzRjYzlkNQ$7el6AzY0mgj51Kjkp9HyuuOCzigiYsCURGQJzwAK+Ec", "26ae729585d28b59c8cba2c0e3cc71fe48a79ccb7097b085eb45e49da34cc9d5"],
		[8, "Test User", "test@sharkmatcher.com", "test", "tester", "$argon2id$v=19$m=65536,t=3,p=4$Njc2YjNjNjdmNTVkMjBlZDlhNDllNDgxZDBmNDM1NDcwYTM5NjIxYzgxZDFkNzA5NjUyYTI1ZmY5ZTU3N2ZhNg$35YlxhaJbVpxhEEa/8F8K8DI+Vnxknn6c+2G3BDtUl8", "676b3c67f55d20ed9a49e481d0f435470a39621c81d1d709652a25ff9e577fa6"],
	]
	write_csv_file(os.path.join(DB_DATA_DIR, "default_users.csv"), users_list, delimiter=",")

	# TODO, allow for export and import actions as CSV files.

def prepare_media(parsed_raw_annotations, annotations_data):
	"""
	Prepare csv file for Media table.
	"""
	row_list = []
	for index, item in enumerate(annotations_data):
		media_name = item[1]
		filename = media_name + ".jpeg"

		# TODO hash the image, not the filename!
		# media_previews[i].hash = SharkMatcher.Util.sha256(media_previews[i].image.src);
		hash = hashlib.sha256(str.encode(filename)).hexdigest()

		# Get associated metadata from raw annotations, if available.
		# This assumes the raw annotations file contains the source of truth of metadata
		# for the images, even though it"s actually directly storing shark metadata.
		# return_dict[image] = (sex, length, tags, nickname, quality, location, image_date_object)
		try:
			metadata = parsed_raw_annotations[media_name]
			sex = metadata["sex"]
			length = metadata["length"]
			location = metadata["location"]
			quality = metadata["quality"]
			datetime = metadata["time"]
			datetime = (time.mktime(datetime.timetuple()))
			row_list.append([index, hash, media_name, filename, index, None, sex, length, location, datetime, "", quality])
		except:
			print("ERROR: Unable to parse for media objects:", media_name)
			row_list.append([index, hash, media_name, filename, index, None, None, None, None, None, "U", None])

	return row_list

def prepare_sharks(parsed_raw_annotations, annotations_data):
	"""
	Prepare csv file for Sharks table and build a shark_reference for future functions.
	"""
	# Build a shark_dict to map sharks with a list of all images associated with each shark.
	# Build a media_dict to map media_ids with media_names.
	shark_dict = {}
	media_dict = {}
	for index, item in enumerate(annotations_data):
		shark_id = item[0]
		media_id = index
		media_dict[index] = item[1]
		
		if shark_dict.get(shark_id) is not None:
			shark_dict[shark_id].append(media_id)
		else:
			shark_dict[shark_id] = [media_id]
	
	# Add every shark to the csv file as either a reference or pointer shark
	row_list = []
	media_shark_reference = {}
	shark_index = 0
	source_index = 0
	for key, value in shark_dict.items():
		for name_index, media_id in enumerate(value):
			try:
				media_name = media_dict[media_id]
				metadata = parsed_raw_annotations[media_name]
				sex = metadata["sex"]
				length = metadata["length"]
				
				# Update the source index at a refrence image (new row).
				if name_index == 0:
					source_index = shark_index
				
				source = source_index

				media_shark_reference[media_name] = shark_index
				row_list.append([shark_index, sex, length, media_id, source])
			except:
				print("ERROR: Unable to parse for shark objects:", name_index, media_id)

			shark_index += 1

	return row_list, media_shark_reference

def prepare_alternateNames(parsed_raw_annotations, annotations_data, media_shark_reference):
	"""
	Prepare csv file for AlternateNames table.
	"""
	index = 0
	row_list = []
	for item in annotations_data:
		media_name = item[1]
		shark = media_shark_reference[media_name]

		try:
			metadata = parsed_raw_annotations[media_name]
			nicknames = metadata["nicknames"]
			if nicknames != "":
				nickname_list = nicknames.split("/")
				for nickname in nickname_list:
					row_list.append([index, shark, nickname, "", ""])
					index += 1
		except:
			print("ERROR: Unable to parse for alternateName objects:", media_name)

	return row_list

def prepare_tags(parsed_raw_annotations, annotations_data, media_shark_reference):
	"""
	Prepare csv file for Tags table.
	"""
	index = 0
	row_list = []
	for item in annotations_data:
		media_name = item[1]
		shark = media_shark_reference[media_name]

		try:
			metadata = parsed_raw_annotations[media_name]
			tags_string = metadata["tags"]
			tags_string = tags_string.lower()
			tags = tags_string.split("/")
			for tag in tags:
				tag_breakdown = tag.split("~")
				# If a tab contains two words, the first represents the tag name, second represents type.
				if len(tag_breakdown) == 2:
					row_list.append([index, shark, tag_breakdown[0], tag_breakdown[1], ""])
					index += 1
				elif len(tag_breakdown) == 1:
					if tag_breakdown[0] != "":
						row_list.append([index, shark, tag_breakdown[0], "unknown", ""])
						index += 1

		except:
			print("ERROR: Unable to parse for tag objects:", media_name)

	return row_list

def main():
	print("Recreating CSV files.")
	# prepare_table_csvs(raw_annotations_file="finMatch_combined_2006-2024.csv", annotations_file=FULL_ANNOTATIONS_PATH, media_directory=IMAGES_DIR)
	prepare_table_csvs(raw_annotations_file="raw_annotations-updated.csv", annotations_file=FULL_ANNOTATIONS_PATH, media_directory=IMAGES_DIR)

	# prepare_table_csvs(raw_annotations_file="database_test_annotations-fixed.csv", annotations_file=FULL_ANNOTATIONS_PATH, media_directory=IMAGES_DIR)

if __name__ == "__main__":
	main()