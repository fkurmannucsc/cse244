import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from backend.constants import DB_DATA_DIR
from backend.constants import OUT_DATA_DIR

import backend.utils
import backend.db.shark
import backend.db.media
import backend.db.action
import backend.db.user
import repo_utils

def db_dump(format):
	"""
	Dump database contents in a specified format.
	"""
	if format == "annotations":
		output_directory = db_dump_annotations()

	if format == "snapshot":
		pass

	
	return output_directory

# TODO, ordering of rows and/or constant IDs.
def db_dump_annotations():
	"""
	Create an annotations file in the Block Lab's format from the current state of the database.
	"""
	annotations_data = []
	max_associations = 0
	quality_dict = {"": 0,
				 	"U": 0,
					"unmatchable": 1,
				 	"poor": 2,
					"medium": 3,
					"good": 4}
	inv_quality_dict = {value: key for key, value in quality_dict.items()}

	all_sharks = backend.db.shark.get_sharks(None, get_full=True, get_source=True, remove_duplicates=True)
	for shark in all_sharks:
		# Fill names field, remove trailing space.
		alternate_names = ""
		for alternate_name in shark["alternate_names"]:
			alternate_names = alternate_names + alternate_name["name"] + "/"
		if (len(alternate_names) != 0):
			alternate_names = alternate_names[:-1]

		# Fill tags field, remove trailing space. TODO, there will be additional tagging information.
		tags = ""
		for tag in shark["tags"]:
			tags = tags + tag["name"] + "~" + tag["type"] + "/"
		if (len(tags) != 0):
			tags = tags[:-1]
		
		# Fill associated media and best media quality.
		associated_medias = []
		max_quality = 0
		for associated_media in shark.associated_medias:
			if not isinstance(associated_media["time"], int):
				associated_media.time = 0
			associated_medias.append((associated_media["name"], associated_media["time"]))
			if associated_media["quality"] is not None and associated_media["quality"] != "":
				quality_value = quality_dict.get(associated_media["quality"])
				if quality_value is None:
					quality_value = 0
				max_quality = max(max_quality, quality_value)

		# Filter by associated media time. This overrides showing the reference image first, which would require no sorting because medias are returned
		# chronologically sorted after the reference image already.
		associated_medias.sort(key=lambda x : x[1])

		# Populate one raw annotations file row.
		shark_row = [associated_medias[0][0], shark.sex, shark.length, tags, alternate_names, inv_quality_dict[max_quality]]
		for media in associated_medias[1:]:
			shark_row.append(media[0])

		max_associations = max(max_associations, len(shark.associated_sharks))
		annotations_data.append(shark_row)

	# Add header row to data.
	header_row = ["id_0", "Sex", "Length", "Tags", "Name", "Best photo quality (good, medium, poor, unmatchable)"]
	for i in range(1, max_associations + 1):
		header = "id_" + str(i)
		header_row.append(header)
	annotations_data.insert(0, header_row)

	# Build output directory and write file.
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	output_directory = os.path.join(OUT_DATA_DIR, "annotations", timestamp)
	os.makedirs(output_directory, exist_ok=True)
	repo_utils.write_csv_file(os.path.join(output_directory, "raw_annotations.csv"), annotations_data)

	# Return the output directory path.
	return output_directory

def db_dump_tables(tables=[], output_directory=None):
	if len(tables) == 0:
		tables.append("all")

	table_data_dict = {}
	
	if ("all" in tables or "Actions" in tables):
		all_rows = backend.db.action.get_actions(ids=None)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.original, row.final, row.action, row.object, row.author, row.time, row.notes]
			table_data.append(table_row)
		table_data_dict["actions"] = table_data

	if ("all" in tables or "Tokens" in tables):
		all_rows = backend.db.user.get_tokens(ids=None)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.token, row.user]
			table_data.append(table_row)
		table_data_dict["tokens"] = table_data

	if ("all" in tables or "Users" in tables):
		all_rows = backend.db.user.get_users(ids=None)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.name, row.email, row.username, row.role, row.password, row.salt]
			table_data.append(table_row)
		table_data_dict["users"] = table_data

	if ("all" in tables or "Labels" in tables):
		all_rows = backend.db.shark.get_associated_objects(ids=None, table="Labels")
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.shark, row.name, row.type, row.time]
			table_data.append(table_row)
		table_data_dict["labels"] = table_data

	if ("all" in tables or "Tags" in tables):
		all_rows = backend.db.shark.get_associated_objects(ids=None, table="Tags")
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.shark, row.name, row.type, row.time]
			table_data.append(table_row)
		table_data_dict["tags"] = table_data

	if ("all" in tables or "AlternateNames" in tables):
		all_rows = backend.db.shark.get_associated_objects(ids=None, table="AlternateNames")
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.shark, row.name, row.type, row.time]
			table_data.append(table_row)
		table_data_dict["alternateNames"] = table_data

	if ("all" in tables or "Sharks" in tables):
		all_rows = backend.db.shark.get_sharks(ids=None, get_full=False, get_source=False, remove_duplicates=False)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.sex, row.length, row.media, row.source]
			table_data.append(table_row)
		table_data_dict["sharks"] = table_data

	if ("all" in tables or "Media" in tables):
		all_rows = backend.db.media.get_medias(ids=None, get_full=False, get_source=False)
		table_data = []
		for row in all_rows:
			table_row = [row.id, row.hash, row.name, row.filename, row.source, row.metadata, row.sex, row.length, row.location, row.time, row.type, row.quality]
			table_data.append(table_row)
		table_data_dict["media"] = table_data

	
		
	# Store output either in it's own directory or add to an existing directory
	if output_directory is None:
		# Build output directory and write file.
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		output_directory = os.path.join(OUT_DATA_DIR, "database-format", timestamp)
		output_directory_main = os.path.join(OUT_DATA_DIR, "database-format")
	os.makedirs(output_directory, exist_ok=True)

	# Write CSVs to output directory.
	for key, value in table_data_dict.items():
		repo_utils.write_csv_file(os.path.join(output_directory, (key + ".csv")), value)
		repo_utils.write_csv_file(os.path.join(output_directory_main, (key + ".csv")), value)

def main():
	db_dump_annotations()
	db_dump_tables("all")

	# db_dump_tables("all", output_directory=os.path.join(OUT_DATA_DIR, "database-format"))

if __name__ == "__main__":
	main()