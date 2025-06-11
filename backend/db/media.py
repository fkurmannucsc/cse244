"""
Media class and database CRUD functions for media objects.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import backend.utils
import backend.db.shark

from backend.constants import SUCCESS
from backend.constants import ERROR

class Media:
    """
    Media class.
    """
    def __init__(self, id=None, hash=None, name=None, filename=None, source=None, shark=None,
                 metadata=None, sex="U", length="U", location=None, time=None, type=None, quality=None):
        """
        Initialize Media object.
        """
        self.id = id
        self.hash = hash
        self.name = name
        self.filename = filename
        self.source = source
        self.shark = shark
        self.metadata = metadata
        self.sex = sex
        self.length = length
        self.location = location
        self.time = time
        self.type = type
        self.quality = quality

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False):
        """
        Return media in dictionary format.
        """
        if isinstance(self.shark, backend.db.shark.Shark):
            self.shark = self.shark.to_dict()
        return {"id": self.id,
                "hash": self.hash,
                "name": self.name,
                "filename": self.filename,
                "source": self.source,
                "shark": self.shark,
                "metadata": self.metadata,
                "sex": self.sex,
                "length": self.length,
                "location": self.location,
                "time": self.time,
                "type": self.type,
                "quality": self.quality}

    def print(self):
        """
        Print object as dict.
        """
        print(self.to_dict())

def db_to_media(db_object):
    """
    Database row to media object.
    """
    if db_object is None:
        return ERROR

    id = db_object[0]
    hash = db_object[1]
    name = db_object[2]
    filename = db_object[3]
    source = db_object[4]
    metadata = db_object[5]
    sex = db_object[6]
    length = db_object[7]
    location = db_object[8]
    time = db_object[9]
    type = db_object[10]
    quality = db_object[11]

    return_object =  Media(id, hash, name, filename, source, None,
                           metadata, sex, length, location, time, type, quality)

    return return_object

# Create/update
def upsert_media(media):
    """
    Update or create media by a media object.
    """
    data = {"id": media.id,
            "hash": media.hash,
            "name": media.name,
            "filename": media.filename,
            "source": media.source,
            "shark": media.shark,
            "metadata": media.metadata,
            "sex": media.sex,
            "length": media.length,
            "location": media.location,
            "time": media.time,
            "type": media.type,
            "quality":media.quality}
    sql = f"""INSERT INTO Media (id, hash, name, filename, source, metadata, sex, length, location, time, type, quality)
    VALUES (:id, :hash, :name, :filename, :source, :metadata, :sex, :length, :location, :time, :type, :quality)
    ON CONFLICT(id) DO UPDATE SET hash = :hash, name = :name, filename = :filename, source = :source,
    metadata = :metadata, sex = :sex, length = :length, location = :location, time = :time, type = :type,
    quality = :quality RETURNING *;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return db_to_media(result[0])

# Getter functions.
def get_media(id, get_full=False, get_source=False):
    """
    Gets media by id. Also gets the shark a media is associated with with get_full.
    Gets source shark media is associated with with get_source.
    """
    data = {"id": id}
    sql = f"""SELECT id, hash, name, filename, source, metadata, sex,
    length, location, time, type, quality FROM Media WHERE id = :id"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    if len(result) == 0:
        return None

    return_media = db_to_media(result[0])

    # TODO get media source.

    # Get media's shark.
    if get_full is True:
        media_ids = get_shark_ids_by_media_ids([id])
        if len(media_ids) == 1:
            return_media.shark = backend.db.shark.get_shark(media_ids[0], get_full=True, get_source=False)

    return return_media

def get_medias(ids, get_full=False, get_source=False, limit=None):
    """
    Gets all media for a list of ids.
    """
    return_list = []

    # If ids is None, get all.
    if (not isinstance(ids, list) or ids is None):
        # print("Getting all.")
        ids = backend.utils.get_all_ids("Media")

    for id in ids:
        media = get_media(id, get_full, get_source)
        if isinstance(media, Media):
            return_list.append(media)

            # If limit is reached, return.
            if limit is not None and len(return_list) == limit:
                break

    return return_list

def get_matching_media_ids(ids=None, names=None, sources=None, sex=None, min_length=None,
                           max_length=None, locations=None, min_time=None, max_time=None,
                           types=None, qualities=None, keywords=None, allow_unknown=False,
                           limit=None):
    """
    Get all media ids, meeting the given conditions.
    """
    # Input validation.
    if sex not in [None, "M", "U", "F"]:
        raise ValueError

    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # Value filters.
    for filter in [("length", min_length, ">=", "min_length"), ("length", max_length, "<=", "max_length"),
                   ("sex", sex, "==", "sex"), ("time", min_time, ">=", "min_time"), ("time", max_time, "<=", "max_time")]:
        if filter[1] is not None:
            # Allow unknown.
            prefix = ""
            if allow_unknown is True:
                prefix = f"{filter[0]} == 'U' OR "

            sql_ids, sql_values = backend.utils.build_sql_placeholders([filter[1]], basename=f"x_{filter[3]}")
            where_clauses.append(f"({prefix}{filter[0]} {filter[2]} {sql_ids})")
            data = data | sql_values

    # List filters.
    for filter in [("id", ids, "id"), ("name", names, "name"), ("source", sources, "source"), ("type", types, "type"),
                    ("quality", qualities, "quality"), ("location", locations, "location"), ("name", keywords, "keyword")]:
        if filter[1] is not None:
            sql_ids, sql_values = backend.utils.build_sql_placeholders(filter[1], basename=f"x_{filter[2]}")
            if filter[2] == "keyword":
                keyword_clause = f"{filter[0]} in {sql_ids} OR "
            else:
                where_clauses.append(f"({filter[0]} in {sql_ids})")
            data = data | sql_values

    # Build where clauses string.
    where_string = ""
    if len(where_clauses) == 0 and keyword_clause != "":
        where_string = " WHERE " + keyword_clause[:-4]
    elif len(where_clauses) != 0:
        where_string = " WHERE " + keyword_clause + backend.utils.list_to_string(where_clauses, " AND ")
    
    # Build the limit string.
    limit_string = ""
    if limit is not None:
        limit_string = f" LIMIT ${limit}"
        data = data | {"limit": limit}

    # Build SQL statement.
    sql = f"""SELECT id FROM Media{where_string}{limit_string};"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return [item[0] for item in result]

# Delete functions.
def delete_media(id):
    """
    Delete media.
    """
    if backend.utils.id_exists(id, "Media") is False:
        return ERROR

    data = {"id": id}
    sql = f"""DELETE FROM Media where id = :id;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    return SUCCESS

# Getter functions.
def get_media_id_by_name(name):
    """
    Gets media id by name.
    """
    data = {"name": name}
    sql = f"""SELECT id FROM Media WHERE name = :name"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    if len(result) == 0:
        return None

    return result[0][0]

# Cross table functions.
def get_shark_ids_by_media_ids(ids, dict_format=False):
    """
    Gets the shark ids associated with a list of media ids. Removes duplicates.
    """
    # Build WHERE clauses and query data.
    where_clauses = []
    data = {}

    # List filters.
    for filter in [("media", ids)]:
        if filter[1] is not None:
            sql_ids, sql_values = backend.utils.build_sql_placeholders(filter[1], basename=f"x_{filter[0]}")
            where_clauses.append(f"({filter[0]} COLLATE NOCASE in {sql_ids})")
            data = data | sql_values

    # Build where clauses string.
    where_string = ""
    if len(where_clauses) != 0:
        where_string = " WHERE " + backend.utils.list_to_string(where_clauses, " AND ")

    # Build SQL statement.
    sql = f"""SELECT id, media FROM Sharks{where_string};"""

    # DB execution.
    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    if dict_format == True:
        result_dict = {}
        for item in result:
            result_dict[item[1]] = item[0]
        return result_dict
    
    result = [(item[0]) for item in result]
    return list(set(result))


def main():
    """
    Main.
    """
    medias = get_medias(ids=[2543, 2526], get_full=True, get_source=False, limit=10)
    for media in medias:
        media.print()

if __name__ == "__main__":
    main()
