#!/bin/bash

# Script to load the sqlite database with data that the db_prepare.py python script loads from various locations in the /data
# directory into prepared CSV files in the /data/database-format directory. By default does not override the Users and Actions table to
# preserve credential information and history.

# Ignore case restrictions.
shopt -s nocasematch

# Help function.
help()
{
    echo "Usage: Run from /shark-identification-matcher: ./db/scripts/populate_db.sh [ -h | --help ]
                [ -a | --actions ]
                [ -d | --database ]
                [ -p | --parse ]
                [ -s | --snapshot ]
                [ -t | --tokens ]
                [ -u | --users ]
                [ -x | --test ]"
    echo
    echo "Populate Shark Matcher Database"

    echo "Options:"
    echo "h     Print this Help."
    echo "a     Update the Actions table. Warning, this will clear all action history."
    echo "d     Drop all database tables and recreate database following latest schema."
    echo "p     Parse data directory and recreate database_format CSV files. Raw data must still be converted separatly."
    echo "s     Repopulate the database from a snapshot previously taken."
    echo "t     Update the Tokens table with default values. Warning, this will revoke all tokens."
    echo "u     Update the Users table with default values. Warning, this will reset all passwords."
    echo "x     Populate database with the contents of annotations files in database-format-test"
}

recreate_csv()
{
    # Run python script to parse data and build database formatted CSV files.
    python3 db/scripts/csv_prepare.py
}

recreate_database()
{
    # Run drop and recreate tables script.
    echo .read db/create_database.sql | sqlite3 db/sharkMatcher.db
}

default_repopulate()
{
    # Drop Tables
    echo "DELETE FROM Labels;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Tags;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM AlternateNames;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Sharks;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Media;" | sqlite3 db/sharkMatcher.db

    # Import csv files into database.
    echo .import --csv data/database-format/sharks.csv Sharks | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format/media.csv Media | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format/alternateNames.csv AlternateNames | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format/tags.csv Tags | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format/labels.csv Labels | sqlite3 db/sharkMatcher.db
}

test_repopulate()
{
    # Drop Tables
    echo "DELETE FROM Labels;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Tags;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM AlternateNames;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Sharks;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Media;" | sqlite3 db/sharkMatcher.db

    # Import csv files into database.
    echo .import --csv data/database-format-test/sharks.csv Sharks | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format-test/media.csv Media | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format-test/alternateNames.csv AlternateNames | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format-test/tags.csv Tags | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format-test/labels.csv Labels | sqlite3 db/sharkMatcher.db
}

snapshot_repopulate()
{
    # Drop Tables
    echo "DELETE FROM Actions;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Tokens;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Users;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Labels;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Tags;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM AlternateNames;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Sharks;" | sqlite3 db/sharkMatcher.db
    echo "DELETE FROM Media;" | sqlite3 db/sharkMatcher.db

    # Import csv files into database.
    echo .import --csv data-out/database-format/sharks.csv Sharks | sqlite3 db/sharkMatcher.db
    echo .import --csv data-out/database-format/media.csv Media | sqlite3 db/sharkMatcher.db
    echo .import --csv data-out/database-format/alternateNames.csv AlternateNames | sqlite3 db/sharkMatcher.db
    echo .import --csv data-out/database-format/tags.csv Tags | sqlite3 db/sharkMatcher.db
    echo .import --csv data-out/database-format/labels.csv Labels | sqlite3 db/sharkMatcher.db
    echo .import --csv data-out/database-format/users.csv Users | sqlite3 db/sharkMatcher.db
    echo .import --csv data-out/database-format/tokens.csv Tokens | sqlite3 db/sharkMatcher.db
    echo .import --csv data-out/database-format/actions.csv Actions | sqlite3 db/sharkMatcher.db
}

actions_repopulate()
{
    # Import finished csv files into database.
    echo "DELETE FROM Actions;" | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format/default_actions.csv Actions | sqlite3 db/sharkMatcher.db
}
tokens_repopulate()
{
    # Import finished csv files into database.
    echo "DELETE FROM Tokens;" | sqlite3 db/sharkMatcher.db
}
users_repopulate()
{
    # Import finished csv files into database.
    echo "DELETE FROM Users;" | sqlite3 db/sharkMatcher.db
    echo .import --csv data/database-format/default_users.csv Users | sqlite3 db/sharkMatcher.db
}

# Count args given.
VALID_ARGUMENTS=$#

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
    default_repopulate
    exit 0
fi

# Get the options. TODO still need to handle long options
while getopts ":hadpstux" opt;
do
    case "$opt" in
        h | --help)
            help
            exit 0
            ;;
        
        a | --actions)
            actions_repopulate
            ;;
        d | --data)
            recreate_database
            ;;
        p | --parse)
            recreate_csv
            ;;
        s | --snapshot)
            snapshot_repopulate
            ;;
        t | --tokens)
            tokens_repopulate
            ;;
        u | --users)
            users_repopulate
            ;;
        x | --test)
            test_repopulate
            ;;
        ?) # Unexpected Option
            echo "Unexpected option: $opt"
            exit 1
            ;;
    esac
done
