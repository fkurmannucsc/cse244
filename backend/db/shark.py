"""
Shark class and database CRUD functions for shark objects.
Shark adjacent classes and CRUD functions for those AssociatedObjects.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import backend.utils
import backend.db
import backend.db.media

from backend.constants import SUCCESS
from backend.constants import ERROR

class Shark:
    """
    Shark class.
    """
    def __init__(self, length="U", sex="U", id=None, source=None, media=None,
                 associated_sharks=[], associated_medias=[], alternate_names=[],
                 labels=[], tags=[]):
        """
        Initialize Shark object.
        """
        self.length = length
        self.sex = sex
        self.id = id
        self.media = media
        self.source = source
        self.associated_sharks = associated_sharks
        self.associated_medias = associated_medias
        self.alternate_names = alternate_names
        self.labels = labels
        self.tags = tags

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False):
        """
        Return a shark in dictionary format.
        """
        # All associated objects to dictionaries.
        associated_medias = [associated_media.to_dict() for associated_media in self.associated_medias]
        associated_sharks = [associated_shark.to_dict() for associated_shark in self.associated_sharks]
        alternate_names = [alternate_name.to_dict() for alternate_name in self.alternate_names]
        labels = [label.to_dict() for label in self.labels]
        tags = [tag.to_dict() for tag in self.tags]

        # If returning compact dict, only return ids of associated objects.
        if compact == True:
            associated_medias = [associated_media["name"] for associated_media in associated_medias]
            associated_sharks = [associated_shark["id"] for associated_shark in associated_sharks]
            alternate_names = [alternate_name["name"] for alternate_name in alternate_names]
            labels = [label["name"] for label in labels]
            tags = [tag["name"] for tag in tags]

        if not hasattr(self, "distance"):
            distance = ""
        else:
            distance = self.distance

        return {"id": self.id,
                "length": self.length,
                "sex": self.sex,
                "source": self.source,
                "media": self.media,
                "associated_sharks": associated_sharks,
                "associated_medias": associated_medias,
                "alternate_names": alternate_names,
                "labels": labels,
                "tags": tags,
                "distance": distance}

    def print(self, compact=False):
        """
        Print object as dict.
        """
        print(self.to_dict(compact))

class AssociatedObject:
    """
    AssociatedObject class.
    """
    def __init__(self, id=None, shark=None, name=None, type=None, time=None):
        """
        Initialize AssociatedObject object.
        """
        self.id = id
        self.shark = shark
        self.name = name
        self.type = type
        self.time = time

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False):
        """
        Represent object as dict.
        """
        return_dict = {"id": self.id,
                        "shark": self.shark,
                        "name": self.name,
                        "type": self.type}

        if self.time is not None and self.time != "":
            return_dict["time"] = self.time

        return return_dict
    def print(self):
        """
        Print object as dict.
        """
        print(self.to_dict())

def db_to_shark(db_object):
    """
    Database row to shark object.
    """
    if db_object is None:
        return ERROR

    id = db_object[0]
    sex = db_object[1]
    length = db_object[2]
    media = db_object[3]
    source = db_object[4]

    return_object = Shark(length, sex, id, source, media)

    return return_object

def db_to_associated_object(db_object):
    """
    Database row to associated object.
    """
    if db_object is None:
        return ERROR

    id = db_object[0]
    shark = db_object[1]
    name = db_object[2]
    type = db_object[3]
    time = db_object[4]

    return_object = AssociatedObject(id, shark, name, type, time)

    return return_object

# Create/update functions.
def upsert_shark(shark):
    """
    Update or create a shark by a shark object.
    """
    data = {"id": shark.id,
            "length": shark.length,
            "sex": shark.sex,
            "media": shark.media,
            "source": shark.source}
    sql = f"""INSERT INTO Sharks (id, length, sex, media, source)
    VALUES (:id, :length, :sex, :media, :source)
    ON CONFLICT(id) DO UPDATE SET length = :length, sex = :sex,
    media = :media, source = :source RETURNING *;
    """
    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    # UPDATE: Don't update the reference images, the users have said that they prefer sharks to revert back 
    # to their pre merge status after being unmerged.
    #
    # If an existing shark is updated get pointer sharks to update as well.
    # if (shark.id is not None and shark.source is not None):
    #     update_shark_ids = shark_get_associated_shark_ids(shark.id)
    #     update_sharks = get_sharks(update_shark_ids, get_full=False, get_source=False)

    #     # Update all sharks needing updates.
    #     for update_shark in update_sharks:
    #         # Update all properties except for the media pointer and source.
    #         data = {"id": update_shark.id,
    #                 "length": shark.length,
    #                 "sex": shark.sex}
    #         sql = f"""UPDATE Sharks SET length = :length, sex = :sex WHERE id = :id;"""
    #         backend.utils.execute_sql(sql, data)

    return db_to_shark(result[0])

def upsert_associated_object(object, table):
    """
    Update or create an associated object of a particular type.
    """
    # Input validation.
    if table not in (backend.utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")
    if backend.utils.id_exists(object.shark, "Sharks") is False:
        return ERROR

    data = {"id": object.id,
            "shark": object.shark,
            "name": object.name,
            "type": object.type,
            "time": object.time}
    sql = f"""INSERT INTO {table} (id, shark, name, type, time) VALUES (:id, :shark, :name, :type, :time)
    ON CONFLICT(id) DO UPDATE SET shark = :shark, name = :name, type = :type, time = :time RETURNING *;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return db_to_associated_object(result[0])

def count_sharks():
    """
    Counts the total number of sharks (also the total number of images because every image is a shark)
    and the total number of source sharks along with how many pointer sharks they each have.
    """
    data = {}

    sql = f"""SELECT COUNT(*) FROM Sharks"""
    total_result = backend.utils.execute_sql(sql, data)

    sql = f"""SELECT source, COUNT(*) FROM Sharks GROUP BY source"""
    group_result = backend.utils.execute_sql(sql, data)

    if total_result == ERROR or group_result == ERROR:
        return ERROR

    return {"groups": group_result, "total": total_result[0][0]}

# Getter functions.
def get_shark(id, get_full=True, get_source=False):
    """
    Gets a shark by id, get_full shark returns all of its associated objects.
    """
    data = {"id": id}
    sql = f"""SELECT id, sex, length, media, source FROM Sharks WHERE id = :id"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    if len(result) == 0:
        return None

    return_shark = db_to_shark(result[0])

    # Get source shark.
    if (get_source is True and return_shark.source != id):
        return get_shark(return_shark.source, get_full, get_source)

    # Get associated object information if get_source, get source's object information,
    # else original target's object information.
    return_shark.alternate_names = []
    return_shark.associated_sharks = []
    return_shark.associated_shark_ids = []
    return_shark.associated_medias = []
    return_shark.associated_media_ids = []
    return_shark.labels = []
    return_shark.tags = []

    if get_full is True:
        # Get all associated objects.
        return_shark.associated_shark_ids = shark_get_associated_shark_ids(id)
        return_shark.alternate_names = get_shark_associated_objects(id, "AlternateNames")
        return_shark.tags = get_shark_associated_objects(id, "Tags")
        return_shark.labels = get_shark_associated_objects(id, "Labels")

        # TODO, decide on whether to return associated_sharks, associated_medias, or just the ids.
        # Get all associated sharks and medias. First item in these lists is the source.
        for shark in return_shark.associated_shark_ids:
            associated_shark = get_shark(id=shark, get_full=False, get_source=False)
            return_shark.associated_media_ids.append(associated_shark.media)
            return_shark.associated_sharks.append(associated_shark)

        return_shark.associated_medias = backend.db.media.get_medias(return_shark.associated_media_ids, get_full=False, get_source=False)

    return return_shark

def get_sharks(ids, get_full=True, get_source=False, remove_duplicates=False, limit=None):
    """
    Gets all sharks for a list of ids.
    """
    return_list = []
    id_tracker = []

    # If ids is None, get all.
    if (not isinstance(ids, list) or ids is None):
        # print("Getting all.")
        ids = backend.utils.get_all_ids("Sharks")

    for id in ids:
        # If getting source, don't re-get, instead track associated sharks already covered.
        if ((get_source is True) and (remove_duplicates is False or id not in id_tracker)):
            shark = get_shark(id, get_full, get_source)

            if isinstance(shark, Shark):
                # Add shark to the return list, if getting source and full shark, can add all associated sharks to id_tracker.
                if (remove_duplicates is False or shark.id not in id_tracker):
                    return_list.append(shark)
                    if (get_source is True and get_full is True):
                        id_tracker = id_tracker + shark.associated_shark_ids
                    else:
                        id_tracker.append(shark.id)
        # Cannot take shortcut if not getting source.
        elif get_source is False:
            shark = get_shark(id, get_full, get_source)
            if isinstance(shark, Shark):
                # Add shark to the return list, if getting source and full shark, can add all associated sharks to id_tracker.
                if (remove_duplicates is False or shark.id not in id_tracker):
                    return_list.append(shark)
                    id_tracker.append(shark.id)

        # If limit is reached, return.
        if limit is not None and len(return_list) == limit:
            break

    return return_list

def get_matching_shark_ids(ids=None, min_length=None, max_length=None, sex=None, medias=None,
                        sources=None, keywords=None, allow_unknown=False, limit=None):
    """
    Get all shark ids meeting the given conditions.
    """
    # Input validation.
    if sex not in [None, "M", "U", "F"]:
        raise ValueError

    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # Value filters.
    for filter in [("length", min_length, ">=", "min_length"),
                   ("length", max_length, "<=", "max_length"), ("sex", sex, "==", "sex")]:
        if filter[1] is not None:
            # Allow unknown.
            prefix = ""
            if allow_unknown is True:
                prefix = f"{filter[0]} == 'U' OR "

            sql_ids, sql_values = backend.utils.build_sql_placeholders([filter[1]], basename=f"x_{filter[3]}")
            where_clauses.append(f"({prefix}{filter[0]} {filter[2]} {sql_ids})")
            data = data | sql_values

    # List filters.
    for filter in [("id", ids, "id"), ("media", medias, "media"),
                   ("source", sources, "source"), ("id", keywords, "keyword")]:
        if filter[1] is not None:
            sql_ids, sql_values = backend.utils.build_sql_placeholders(filter[1], basename=f"x_{filter[2]}")
            if filter[2] == "keyword":
                keyword_clause = f"{filter[0]} COLLATE NOCASE in {sql_ids} OR "
            else:
                where_clauses.append(f"({filter[0]} COLLATE NOCASE in {sql_ids})")
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
    sql = f"""SELECT id FROM Sharks{where_string}{limit_string};"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return [item[0] for item in result]

def count_associated_objects(table=None):
    """
    Counts the total number of associated objects for a specific table.
    """
    # Input validation.
    if table not in (backend.utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")

    data = {}
    sql = f"""SELECT COUNT(*) FROM {table}"""
    result = backend.utils.execute_sql(sql, data)

    if result == ERROR:
        return ERROR

    return result[0][0]

def get_associated_object(id=None, name=None, table=None):
    """
    Get an associated object from a specified associated object table, get by id or name supported.
    """
    # Input validation.
    if table not in (backend.utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")

    if id is None:
        query = "name"
        data = {"name": name}
    else:
        query = "id"
        data = {"id": id}

    sql = f"""SELECT id, shark, name, type, time FROM {table} WHERE {query} = :{query}"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    if len(result) == 0:
        return None

    return db_to_associated_object(result[0])

def get_associated_objects(ids, limit=None, table=None):
    """
    Gets all associated objects for a list of ids and table.
    """
    # Input validation.
    if table not in (backend.utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")
    
    return_list = []

    # If ids is None, get all.
    if (not isinstance(ids, list) or ids is None):
        # print("Getting all.")
        ids = backend.utils.get_all_ids(table)

    for id in ids:
        object = get_associated_object(id=id, table=table)
        if isinstance(object, AssociatedObject):
            return_list.append(object)

            # If limit is reached, return.
            if limit is not None and len(return_list) == limit:
                break

    return return_list

# TODO, non exact search for names.
def get_matching_associated_object_sharks(ids=None, sharks=None, names=None, types=None,
                                          keywords=None, allow_unknown=False, table=None,
                                          limit=None):
    """
    Get all associated object ids of a table, meeting the given conditions.
    """
    # Input validation.
    if table not in (backend.utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")

    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # List filters.
    for filter in [("id", ids, "id"), ("shark", sharks, "shark"), ("name", names, "name"),
                   ("type", types, "type"), ("name", keywords, "keyword")]:
        if filter[1] is not None:
            sql_ids, sql_values = backend.utils.build_sql_placeholders(filter[1], basename=f"x_{filter[2]}")
            if filter[2] == "keyword":
                keyword_clause = f"{filter[0]} COLLATE NOCASE in {sql_ids} OR "
            else:
                where_clauses.append(f"({filter[0]} COLLATE NOCASE in {sql_ids})")
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
    sql = f"""SELECT shark FROM {table}{where_string}{limit_string};"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return [item[0] for item in result]

# Delete functions.
def delete_shark(id):
    """
    Delete a shark. Does not automatically delete Tags and AlternateNames associated with the deleted shark.
    If a source shark is deleted, the pointer sharks are reassigned.
    """
    # Check if shark id exists
    if backend.utils.id_exists(id, "Sharks") is False:
        return ERROR

    # Get source shark.
    delete_object = get_shark(id, get_full=False, get_source=False)
    source_id = delete_object.source

    # Edge case where shark to delete is source the shark, reassign pointer sharks first.
    if source_id == id:
        updated_source_id = reassign_pointer_sharks(delete_object)

        # If new_source_id is same as old source_id, then this is the last associated shark,
        # the whole shark will be deleted and a source_id of None should be returned.
        if updated_source_id == source_id:
            updated_source_id = None

        source_id = updated_source_id

    data = {"id": id}
    sql = f"""DELETE FROM Sharks where id = :id;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return SUCCESS, source_id

def delete_associated_object(id, table):
    """
    Delete an associated object from a specified associated object table.
    """
    if table not in (backend.utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")
    if backend.utils.id_exists(id, table) is False:
        return ERROR

    data = {"id": id}
    sql = f"""DELETE FROM {table} where id = :id;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return SUCCESS

# Matching functions.
def reassign_pointer_sharks(old_source_shark, new_source_shark=None):
    """
    Given an old source shark, point that shark and all sharks pointing to it to point a new source.
    If new source shark is none, sets the new source shark to be the associated shark of old
    source shark with the oldest media object.
    Returns ID of the new source shark.
    """
    print("Reassigning pointer sharks.")
    # Check if shark id exists.
    if (backend.utils.id_exists(old_source_shark.id, "Sharks") is False
        or (new_source_shark is not None and backend.utils.id_exists(old_source_shark.id, "Sharks") is False)):
        return ERROR

    # Get pointer sharks.
    update_shark_ids = shark_get_associated_shark_ids(old_source_shark.id)
    update_sharks = get_sharks(update_shark_ids, get_full=False, get_source=False)
    for update_shark in update_sharks:
        print(f"Update shark: {update_shark.id}, {update_shark.length}.")

    # Check that old_source_shark is a source shark with pointer sharks.
    if (old_source_shark.source != old_source_shark.id or len(update_shark_ids) == 0):
        raise ValueError("Invalid source shark given.")

    # Get/check new_source_shark.
    # If not given as parameter or invalid, default to oldest image shark.
    # If already pointing to oldest image shark, point to next oldest unless there is only one.
    if new_source_shark is None:
        update_sharks = sort_sharks(update_sharks, "media_time")
        if ((update_sharks[0].id != old_source_shark.id) or (len(update_sharks) == 1)):
            new_source_shark = update_sharks[0]
        else:
            new_source_shark = update_sharks[1]

    # Update all sharks needing updates.
    for update_shark in update_sharks:
        # Update all properties except for the media pointer.
        # data = {"id": update_shark.id,
        #         "length": new_source_shark.length,
        #         "sex": new_source_shark.sex,
        #         "source": new_source_shark.id}
        # sql = f"""UPDATE Sharks SET length = :length, sex = :sex,
        # source = :source WHERE id = :id;"""

        # Update just the source property, so that when you unmerge, you return to the shark's 
        # properties prior to the merge.
        print(f"Updating pointer shark: {update_shark.id}, {update_shark.length}.")
        data = {"id": update_shark.id,
                "source": new_source_shark.id}
        sql = f"""UPDATE Sharks SET source = :source WHERE id = :id;"""

        result = backend.utils.execute_sql(sql, data)
        if result == ERROR:
            return ERROR

    # Update associated objects.
    for update_shark in update_sharks:
        for table in ["AlternateNames", "Tags", "Labels"]:
            update_objects = get_shark_associated_objects(update_shark.id, table)
            for object in update_objects:
                object.shark = new_source_shark.id
                upsert_associated_object(object, table)

    update_sharks = get_sharks(update_shark_ids, get_full=False, get_source=False)
    for update_shark in update_sharks:
        print(f"FINAL Update shark: {update_shark.id}, {update_shark.length}.")

    return new_source_shark.id

def sort_sharks(shark_list, parameter="id"):
    """
    Given a list of sharks, sort in ascending order by the given parameter.
    """
    print("Sharks", shark_list)
    for shark in shark_list:
            shark.print()
    # Sort on shark attributes
    if parameter == "id":
        shark_list.sort(key=lambda x : x.id)
    elif parameter == "length":
        shark_list.sort(key=lambda x : x.length)

    # Sort on media attributes
    elif parameter == "media_time":
        shark_media_list = [(shark, shark.media, None) for shark in shark_list]
        media_ids = [shark.media for shark in shark_list]

        # Get media data to sort on.
        medias = backend.db.media.get_medias(media_ids)
        print("medias")
        for media in medias:
            media.print()
        shark_media_list = []
        for index, item in enumerate(shark_list):
            # Catch malformed media.
            if not isinstance(medias[index].time, int):
                shark_media_list.append((item, 0))
            else:
                shark_media_list.append((item, medias[index].time))

        # Sort by media, rebuild shark_list.
        shark_media_list.sort(key=lambda x : x[1])
        shark_list = [item[0] for item in shark_media_list]

    return shark_list

# Cross table functions.
def get_shark_associated_objects(id, table):
    """
    Get associated objects for a shark id from a specific table.
    """
    # Input validation.
    if table not in (backend.utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")

    data = {"id": id}
    sql = f"""SELECT id, shark, name, type, time FROM {table} WHERE shark = :id;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return [db_to_associated_object(item) for item in result]

def associated_object_ids_get_associated_shark_ids(ids, table):
    """
    Get all associated shark ids for a list of associated objects.
    """
    if table not in (backend.utils.TABLE_NAMES):
        raise ValueError("Error, invalid database table.")
    
    # If ids is None, get all.
    if (not isinstance(ids, list) or ids is None):
        # print("Getting all.")
        ids = backend.utils.get_all_ids(table)

    sql_ids, sql_values = backend.utils.build_sql_placeholders(ids)
    data = {**sql_values, **sql_values} # Can use this syntax to add another sqlvalues object to this dictionary.
    sql = f"""SELECT shark FROM {table} WHERE id IN {sql_ids};"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return [item[0] for item in result]

def shark_get_associated_shark_ids(id):
    """
    Get all associated shark ids for a shark. The source shark ID is always the first in the list.
    """
    # Check if id exists.
    if backend.utils.id_exists(id, "Sharks") is False:
        return ERROR

    # Get the source shark, if shark_id is not already one.
    source_shark = get_shark(id, get_full=False, get_source=True)

    data = {"source": source_shark.id}
    sql = f"""SELECT id FROM Sharks WHERE source = :source"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    # Place the source at the beginning of the result list.
    result_list = [item[0] for item in result]
    result_list.remove(source_shark.id)
    result_list.insert(0, source_shark.id)

    return result_list

def main():
    """
    Main.
    """
    sharks = get_sharks(ids=[2135, 9], get_full=True, get_source=False, remove_duplicates=False, limit=10)
    for shark in sharks:
        shark.print()
    
    reassign_pointer_sharks(sharks[0], sharks[1])

    sharks = get_sharks(ids=[2135], get_full=True, get_source=False, remove_duplicates=False, limit=10)
    for shark in sharks:
        shark.print()



if __name__ == "__main__":
    main()
