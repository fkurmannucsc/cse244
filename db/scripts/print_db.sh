#!/bin/bash

# Script to load the entire database from nothing - update data by: updating files following same convention,
# running this script, database is repopulated to match files.
pwd

# Print all tables in the database.
echo Printing all tables:
echo .tables | sqlite3 db/sharkMatcher.db

# Print all schemas in the database.
echo Printing all schemas:
echo .schema Media | sqlite3 db/sharkMatcher.db
echo .schema Sharks | sqlite3 db/sharkMatcher.db
echo .schema AlternateNames | sqlite3 db/sharkMatcher.db
echo .schema Tags | sqlite3 db/sharkMatcher.db
echo .schema Labels | sqlite3 db/sharkMatcher.db
echo .schema Users | sqlite3 db/sharkMatcher.db
echo .schema Tokens | sqlite3 db/sharkMatcher.db
echo .schema Actions | sqlite3 db/sharkMatcher.db

# Print first 5 entires of each table.
echo Printing Heads
echo
echo Media:
echo "SELECT COUNT(*) from Media;" | sqlite3 db/sharkMatcher.db
echo "SELECT * from Media LIMIT 5;" | sqlite3 db/sharkMatcher.db
echo
echo Sharks:
echo "SELECT COUNT(*) from Sharks;" | sqlite3 db/sharkMatcher.db
echo "SELECT * from Sharks LIMIT 5;" | sqlite3 db/sharkMatcher.db
echo
echo AlternateNames:
echo "SELECT COUNT(*) from AlternateNames;" | sqlite3 db/sharkMatcher.db
echo "SELECT * from AlternateNames LIMIT 5;" | sqlite3 db/sharkMatcher.db
echo
echo Tags:
echo "SELECT COUNT(*) from Tags;" | sqlite3 db/sharkMatcher.db
echo "SELECT * from Tags LIMIT 5;" | sqlite3 db/sharkMatcher.db
echo
echo Labels:
echo "SELECT COUNT(*) from Labels;" | sqlite3 db/sharkMatcher.db
echo "SELECT * from Labels LIMIT 5;" | sqlite3 db/sharkMatcher.db
echo
echo Users:
echo "SELECT COUNT(*) from Users;" | sqlite3 db/sharkMatcher.db
echo "SELECT * from Users LIMIT 5;" | sqlite3 db/sharkMatcher.db
echo
echo Tokens:
echo "SELECT COUNT(*) from Tokens;" | sqlite3 db/sharkMatcher.db
echo "SELECT * from Tokens LIMIT 5;" | sqlite3 db/sharkMatcher.db
echo
echo Actions:
echo "SELECT COUNT(*) from Actions;" | sqlite3 db/sharkMatcher.db
echo "SELECT * from Actions LIMIT 5;" | sqlite3 db/sharkMatcher.db

