"""
User class and database CRUD functions for user objects.
"""

import hashlib
import hmac
import os
import random
import sys

from argon2 import PasswordHasher

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

import backend.utils

from backend.utils import SUCCESS
from backend.utils import ERROR

class User:
    """
    User class.
    """
    def __init__(self, id=None, name=None, email=None, username=None,
                 role=None, password=None, salt=None):
        """
        Initialize User object.
        """
        self.id = id
        self.name = name
        self.email = email
        self.username = username
        self.role = role
        self.password = password
        self.salt = salt

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False):
        """
        Return a user in dictionary format.
        """
        return {"id": self.id,
                "name": self.name,
                "email": self.email,
                "username": self.username,
                "role": self.role,
                "password": self.password,
                "salt": self.salt}
    def to_secure_dict(self):
        """
        Return a user in dictionary format.
        """
        return {"id": self.id,
                "name": self.name,
                "email": self.email,
                "username": self.username,
                "role": self.role}

    def print(self):
        """
        Print a user.
        """
        print(self.to_dict())

def db_to_user(db_object):
    """
    Database row to user object.
    """
    if db_object is None:
        return ERROR

    id = db_object[0]
    name = db_object[1]
    email = db_object[2]
    username = db_object[3]
    role = db_object[4]
    password = db_object[5]
    salt = db_object[6]

    return User(id, name, email, username, role, password, salt)

class Token:
    """
    User class.
    """
    def __init__(self, id=None, token=None, user=None):
        """
        Initialize Token object.
        """
        self.id = id
        self.token = token
        self.user = user

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self, compact=False):
        """
        Return a token in dictionary format.
        """
        return {"id": self.id,
                "token": self.token,
                "user": self.user}

    def print(self):
        """
        Print a token.
        """
        print(self.to_dict())

def db_to_token(db_object):
    """
    Database row to token object.
    """
    if db_object is None:
        return ERROR

    id = db_object[0]
    token = db_object[1]
    user = db_object[2]

    return Token(id, token, user)

# Create/update functions.
def upsert_user(user):
    """
    Update or create a user by user object.
    """

    data = ({"id": user.id,
            "name": user.name,
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "password": user.password,
            "salt": user.salt})
    sql = f"""INSERT INTO Users (id, name, email, username, role, password, salt)
    VALUES (:id, :name, :email, :username, :role, :password, :salt)
    ON CONFLICT(id) DO UPDATE SET name = :name, email = :email, 
    username = :username, role = :role, password = :password, salt = :salt
    RETURNING *;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return db_to_user(result[0])

def upsert_token(token):
    """
    Update or create a token by token object.
    """
    data = ({"id": token.id,
            "token": token.token,
            "user": token.user})
    sql = f"""INSERT INTO Tokens (id, token, user) VALUES (:id, :token, :user)
    ON CONFLICT(id) DO UPDATE SET token = :token, user = :user RETURNING *;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return db_to_token(result[0])

# Getter functions.
def get_user(id=None, username=None):
    """
    Get a user by username or by id.
    """
    # Get by username.
    if id is None:
        query = "username"
        data = {"username": username}
    # Get by id.
    else:
        query = "id"
        data = {"id": id}

    sql = f"""SELECT id, name, email, username, role, password, salt
    FROM Users WHERE {query} = :{query}"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    if len(result) == 0:
        return None

    return db_to_user(result[0])

def get_users(ids):
    """
    Gets all users for a list of ids.
    """
    return_list = []

    # If ids is None, get all.
    if (not isinstance(ids, list) or ids is None):
        ids = backend.utils.get_all_ids("Users")

    for id in ids:
        user = get_user(id)
        if isinstance(user, User):
            return_list.append(user)

    return return_list

# TODO
def get_matching_users(ids=None, names=None, sources=None, sex=None, 
                       min_length=None, max_length=None, locations=None,
                        min_time=None, max_time=None, types=None, qualities=None,
                        allow_unknown=False):
    """
    Get all users, meeting the given conditions.
    """
    # Input validation.
    if sex not in [None, "M", "U", "F"]:
        raise ValueError

    # Build WHERE clauses and query data.
    where_clauses = []
    data = {}

    # Value filters.
    for filter in [("length", min_length, ">=", "min_length"), ("length", max_length, "<=", "max_length"),
                   ("sex", sex, "==", "sex"), ("time", min_time, ">=", "min_time"),
                   ("time", max_time, "<=", "max_time")]:
        if filter[1] is not None:
            # Allow unknown.
            prefix = ""
            if allow_unknown is True:
                prefix = f"{filter[0]} == 'U' OR "

            sql_ids, sql_values = backend.utils.build_sql_placeholders([filter[1]],basename=f"x_{filter[3]}")
            where_clauses.append(f"({prefix}{filter[0]} {filter[2]} {sql_ids})")
            data = data | sql_values

    # List filters.
    for filter in [("id", ids), ("name", names), ("source", sources),
                   ("type", types), ("quality", qualities), ("location", locations)]:
        if filter[1] is not None:
            sql_ids, sql_values = backend.utils.build_sql_placeholders(filter[1], basename=f"x_{filter[0]}")
            where_clauses.append(f"({filter[0]} in {sql_ids})")
            data = data | sql_values

    # Build where clauses string.
    where_string = ""
    if len(where_clauses) != 0:
        where_string = " WHERE " + backend.utils.list_to_string(where_clauses, " AND ")

    # Build SQL statement.
    sql = f"""SELECT id, hash, name, source, metadata, sex,
           length, location, time, type, quality FROM Media{where_string};"""

    result = backend.utils.execute_sql(sql, data)
    return [item[0] for item in result]

def get_token(id=None, token=None):
    """
    Get a token by id or by token.
    """
    # Get by token.
    if id is None:
        query = "token"
        data = {"token": token}
    # Get by id.
    else:
        query = "id"
        data = {"id": id}

    sql = f"""SELECT id, token, user FROM Tokens WHERE {query} = :{query}"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR
    if len(result) == 0:
        return None

    return db_to_token(result[0])

def get_tokens(ids):
    """
    Gets all tokens for a list of ids.
    """
    return_list = []

    # If ids is None, get all.
    if (not isinstance(ids, list) or ids is None):
        ids = backend.utils.get_all_ids("Tokens")

    for id in ids:
        token = get_token(id)
        if isinstance(token, Token):
            return_list.append(token)

    return return_list

# Delete functions.
def delete_user(id):
    """
    Delete a user.
    """
    if backend.utils.id_exists(id, "Users") is False:
        return ERROR

    data = {"id": id}
    sql = f"""DELETE FROM Users where id = :id;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return SUCCESS

def delete_token(id):
    """
    Delete a token.
    """
    if backend.utils.id_exists(id, "Tokens") is False:
        return ERROR

    data = {"id": id}
    sql = f"""DELETE FROM Tokens where id = :id;"""

    result = backend.utils.execute_sql(sql, data)
    if result == ERROR:
        return ERROR

    return SUCCESS

# Authentication functions.
def authenticate_user(username, token):
    """
    Return True if authenticated or False if not.
    """
    # Get token and user from database.

    user = get_user(username=username)
    if not isinstance(user, User):
        return False

    ph = PasswordHasher()
    token_hash = ph.hash(token, salt=user.salt.encode("utf-8"))
    returned_token = get_token(token=token_hash)
    if not isinstance(returned_token, Token):
        return False

    # If the token and user don't match, return False.
    if returned_token.user != user.id:
        return False

    # Success
    return True

def login_user(username, password):
    """
    Check if username/password combo is correct and if so return a user/token object.
    """
    user = get_user(username=username)
    if not isinstance(user, User):
        return ERROR

    # Cryptographically hash the password with the salt.
    ph = PasswordHasher()
    hash = ph.hash(password, salt=user.salt.encode("utf-8"))

    # If user is authenticated, generate a token, return the username/token combo.
    if hmac.compare_digest(user.password, hash) == True:
    # if user.password == hash:
        # Generate token, store in tokens table
        token = random.random()
        token = hashlib.sha256(str.encode(str(token))).hexdigest()
        token_hash = ph.hash(token, salt=user.salt.encode("utf-8"))
        generated_token = Token(None, token_hash, user.id)
        token_status = upsert_token(generated_token)
        if token_status == ERROR:
            print("Issue saving token.")
            return ERROR

        user_dict = user.to_secure_dict()
        user_dict["token"] = token

        return user_dict

def logout_user(username, token):
    """
    Revoke a token, return success or error.
    """
    # Get token and user from database.
    user = get_user(username=username)

    # If this is not a valid user, there is no token, just log out.
    if not isinstance(user, User):
        return SUCCESS

    ph = PasswordHasher()
    token_hash = ph.hash(token, salt=user.salt.encode("utf-8"))
    returned_token = get_token(token=token_hash)
    if not isinstance(returned_token, Token):
        print("Warning, no token to delete.")
        return SUCCESS

    status = delete_token(returned_token.id)

    return status

def update_user_password(username, password):
    # Get the original user object.
    user = get_user(username=username)
    if not isinstance(user, User):
        return ERROR

    # Cryptographically hash the password with the salt.
    ph = PasswordHasher()
    hash = ph.hash(password, salt=user.salt.encode("utf-8"))

    user.password = hash
    updated_user = upsert_user(user)
    if updated_user == ERROR:
        return ERROR

    return SUCCESS

# Main
def main():
    """
    Main.
    """

if __name__ == "__main__":
    main()
