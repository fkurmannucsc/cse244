"""
Action class and database CRUD functions for action objects.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import backend.utils

from backend.utils import SUCCESS
from backend.utils import ERROR

class Action:
    """
    Action class.
    """
    def __init__(self, id=None, original="", final="", action="", object="", author=None, time=0, notes=None):
        """
        Initialize Action object.
        """
        self.id = id
        # Note for original and final, filtering on "" gets enpty originals/finals.
        self.original = original
        self.final = final
        self.action = action
        self.object = object
        self.author = author
        self.time = time
        self.notes = notes

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False):
        """
        Return an action in dictionary format.
        """
        return {"id": self.id,
                "original": self.original,
                "final": self.final,
                "action": self.action,
                "object": self.object,
                "author": self.author,
                "time": self.time,
                "notes": self.notes}

    def print(self):
        """
        Print an action.
        """
        print(self.to_dict())

def db_to_action(db_object):
    """
    Database row to action object.
    """
    if db_object is None:
        return ERROR

    id = db_object[0]
    original = db_object[1]
    final = db_object[2]
    action = db_object[3]
    object = db_object[4]
    author = db_object[5]
    time = db_object[6]
    notes = db_object[7]

    return_object = Action(id, original, final, action, object, author, time, notes)

    return return_object

# Create/update functions.
def upsert_action(action):
    """
    Update or create a action by action object.
    """
    data = ({"id": action.id,
            "original": action.original,
            "final": action.final,
            "action": action.action,
            "object": action.object,
            "author": action.author,
            "time": action.time,
            "notes": action.notes})
    sql = f"""INSERT INTO Actions (id, original, final, action, object, author, time, notes)
    VALUES (:id, :original, :final, :action, :object, :author, :time, :notes)
    ON CONFLICT(id) DO UPDATE SET original = :original, final = :final, action = :action, 
    object = :object, author = :author, time = :time, notes = :notes RETURNING *;
    """

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    return db_to_action(result[0])

# Getter functions.
def get_action(id):
    """
    Get a action by id.
    """

    data = {"id": id}
    sql = f"""SELECT id, original, final, action, object, author, time, notes
    FROM Actions WHERE id = :id"""

    result = backend.utils.execute_sql(sql, data)[0]
    if result == ERROR:
        return ERROR
    if len(result) == 0:
        return None

    return db_to_action(result)

def get_actions(ids, limit=None):
    """
    Gets all actions for a list of ids.
    """
    return_list = []

    # If ids is None, get all.
    if (not isinstance(ids, list) or ids is None):
        # print("Getting all.")
        ids = backend.utils.get_all_ids("Actions")

    for id in ids:
        action = get_action(id)
        if isinstance(action, Action):
            return_list.append(action)

            # If limit is reached, return.
            if limit is not None and len(return_list) == limit:
                break

    return return_list

# TODO keywords search.
def get_matching_action_ids(ids=None, original=None, object=None, final=None, author=None, min_time=None,
                            max_time=None, action=None, keywords=None, allow_unknown=False, limit=None):
    """
    Get all medias, meeting the given conditions.
    """
    # Input validation.

    # Build WHERE clauses and query data.
    where_clauses = []
    keyword_clause = ""
    data = {}

    # Value filters.
    for filter in [("action", action, "==", "action"), ("author", author, "==", "author"),
                   ("original", original, "==", "original"), ("object", object, "==", "object"),
                   ("final", final, "==", "final"), ("time", min_time, ">=", "min_time"),
                   ("time", max_time, "<=", "max_time")]:
        if filter[1] is not None:
            # Allow unknown.
            prefix = ""

            sql_ids, sql_values = backend.utils.build_sql_placeholders([filter[1]], basename=f"x_{filter[3]}")
            where_clauses.append(f"({prefix}{filter[0]} {filter[2]} {sql_ids})")
            data = data | sql_values

    # List filters.
    for filter in [("id", ids, "id")]:
        if filter[1] is not None:
            sql_ids, sql_values = backend.utils.build_sql_placeholders(filter[1], basename=f"x_{filter[2]}")
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
    sql = f"""SELECT id FROM Actions{where_string}{limit_string};"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    return [item[0] for item in result]

# Delete functions.
def delete_action(id):
    """
    Delete an action.
    """
    if backend.utils.id_exists(id, "Actions") is False:
        return ERROR

    data = {"id": id}
    sql = f"""DELETE FROM Actions where id = :id;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    return SUCCESS

# Main
def main():
    """
    Main.
    """

if __name__ == "__main__":
    main()
