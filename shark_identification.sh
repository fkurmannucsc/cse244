#!/bin/bash

# Ignore case restrictions.
shopt -s nocasematch

# Help function.
help()
{
  echo "Usage: ./run.sh [ -h | --help ]
                [ -a | --annotations ]
                [ -s | --snapshot ]
                [ -e | --embeddings ]
                [ -r | --run]
                [ -t | --test]"
  echo
  echo "Shark Identification:"

  echo "Options:"
  echo "h     Print this Help."
  echo "a     Read in an annotations file. Populate the database, create corresponding train and validation data files, 
              then populate the images directory."
  echo "s     Read in data from a snapshot. Populate the database, create corresponding train and validation data files, 
              then populate the images directory."
  echo "e     Given the data in the database and the annotations files present, generate embeddings."
  echo "r     Run the server without updating the data or embeddings."
  echo "t     Run the server with a toy embeddings, for testing only."
}

# Annotations function.
annotations()
{
  echo "Storing annotations data."

}

# Snapshot function.
snapshot()
{
  echo "Storing snapshot data."

}

# Run function.
run()
{
  echo "Running Shark Matcher server."
  python3 frontend/cli/server.py . --no-browser
}

# Run function.
test_run()
{
  echo "Running Shark Matcher server with test data."
  python3 frontend/cli/server.py . --no-browser --model test
}

# NEW run function.
embeddings()
{
  echo "Generating embeddings."

}

# Count args given.
VALID_ARGUMENTS=$#

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
  help
  exit 0
fi

# Get the options. TODO still need to handle long options
while getopts ":hasert" opt;
do
    case "$opt" in
        h | --help)
            echo "Help called"
            help
            exit 0
            ;;
        a | --annotations)
            annotations
            exit 0
            ;;
        s | --snapshot)
            snapshot
            exit 0
            ;;
        e | --embeddings)
            embeddings
            exit 0
            ;;
        r | --run)
            run
            exit 0
            ;;
        t | --test)
            dev_run
            exit 0
            ;;
        ?) # Unexpected Option
            echo "Unexpected option: $opt"
            exit 1
            ;;
    esac
done
