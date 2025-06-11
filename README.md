# shark-matcher

## Directory Structure

### /api
Contains api references and rules.

### /backend
All code responsible for data processing, retrieval, and handling database operations.
- /db - All database CRUD functionality.
- /model - Data loadeing and retrieval code; these files interact directly with trained model data saved as .pt files.
- utils.py - General utility functions used throughout the backend directory.
- constants.py - Constants used throughout the backend directory.
- analyze_data.py - Data analysis utilities for data cleaning and summarization.
- convert_data.py - Parses data in the data/raw-data directory and produces parsed annotations files and images for use by the
database and frontend.

### /frontend
All code files including html, css, and javascript to display and run UI as well as python API handler files.
- /cli - Interface to start the server via the command line
- /static - Browser relevant code including style sheets, images and files accessed by the GUI, and JS files.
- /static/js/modules/sharkMatcher - Client side API code including endpoints, request and response handlers.
- /static/js/modules/webui - GUI handling code reponsible for page rendering, navigation and event handling.
- handlers.py - API request helper functions, handle the core of requests.
- server.py - Python server code and server side API endpoint definitions.
- common.py - Common utility functions.
- *.py - API handler scripts for corresponding object types.

### /data
All data in both raw format received parsed format, also stores trained models for use by Shark Matcher
- /annotations - Parsed annotations files (tsv files storing metadata and match information on each image). These files are read by model training and retrieval scripts and used to populate the database.
- /database-format. CSV files containing data formatted according to the database schema for loading into the database. These 
files are produced by /db/scripts reading data in this /data directory.
- /embeddings - Pytorch (.pt) embeddings files built via the shark-identification-model.
- /images - Flat directory containing all parsed images.
- /raw-data - Unparsed matching spreadsheet csv files and unparsed images in nested directory structure as received from Block lab or other original source.

### /data_out
Directory where database snapshots, logs, and output files appear.

### /db
Database directory with database schema, files, population and snapshot scripts.
- /scripts - Scripts for database population, snapshots, and debugging.
- create_database.sql - Database schema.
- sharkMatcher.db - Sqlite database file.

### /tests
Testing directory with tests for functionality and style.
- /data - Small dataset for testing.
- python_style_tests.py - Pylint style testing script.
- run_tests.sh - Script to allow running a desired set of tests.
- *.py - Tests for corresponding python files.


## Workflow
#### Prerequisites TODO:
- Ensure dependencies (`requirements.txt`).
    pip install -r requirements.txt

1. Clone or pull the latest version of the shark-matcher repository.
2. Assure data directory is populated with all the necessary files and structure outlined in the **data** section above.

- Model files stored in data/embeddings
- Parsed annotations in the /data/annotations directory. Parsed annotations can be produced from clean raw-data with the convert_data script. `python3 backend/convert_data.py `.

4. Populate the database with the parsed data. `./db/scripts/populate_db.sh [-h for info]`.
- `./db/scripts/print_db.sh` can print a preview of the database to verify.

#### Running the Server:
To run the server, the database needs to be populated, formatted annotations files need to be present in data/annotations, and embeddings and neighbors files must be stored in data/embeddings. All images referenced in the 
database must also be present in the frontent/static/images directory.

After having assured the data and embeddings dependencies above, run the server from /shark-matcher with:
    `./shark_identification.sh -r`

#### Model Training:

1. Make sure that all necessary images are located in the data/raw-data/images directory and that the data/images directory is empty or free of old/irrelevant data. Placing new images in raw-data/images is needed so they can be parsed and converted by the convert_data script which copies them to the data/imges directory in the next step.
2. Format the images and populate the formatted image directory data/images by running the format_image_directory() method in convert_data.py.
3. Populate the database with data from a raw annotations file or database snapshot by running ./db/scripts/populate_db.sh. See the options for snapshot vs annotations population.
4. With the database populated you can generate the formatted annotations files by running the prepare_dataset() function in convert_data.py.

You should now have the images, the database, and the tsv annotations files in their respective locations and should be ready to begin training. Training will then create SharkfinDataset objects for the train and test sets based on the splits in the annotations files, which you created by running prepare_dataset.
At this point, make sure your configurations are correct in config.py. Then to start training run:

python3 backend/learning/trainer/lora_trainer.py


#### Embeddings Generation
Once you have trained a model and it is present in the backend/models/trained/lora directory (this includes all 6 items created during model training - 2 directories and 4 files) you are ready to generate embeddings using the trained model.

To generate embeddings, store neighbors, and prepare for evaluation of this model on your dataset, you will need to make sure that the database is populated, annotations files are populated, and all referenced images are present in the data/images directory. This requires the same process as steps 1, 2, 3, 4 from the model training process above.

Finally, for the embeddings to be generated, a SharkfinDataset object must be cached in a pt file. This can be done by running the store_dataset() function in backend/retrieval/embeddings.py. You should now have a stored_dataset.pt file in data/embeddings. This dataset is always made by reading the FULL_ANNOTATIONS file since likely you'll want to create embeddings for all your images, not restricting to just the ones in the train/test sets.

You should now have the images, annotations file, the database, and a stored pt file of the database in their respective locations and should be ready to begin embeddings generation.

To store embeddings for a dataset run the store_embeddings() function in backend/retrieval/embeddings.py. Storing centroid embeddings is required for the different retrieval options, train/test split is optional and will split train and test images to not belong to the same centroids for centroid embeddings. The train/test split parameter therefore has no effect on the generation of regular image embeddings.

#### Embeddings Evaluation
Evaluate model's embedding generate. Evaluate retrieval methods. 

Another insight is that with these evaluation scripts, you are able to quickly find queries that the matching algorithm struggles on consistently. In some but not all cases, these are model limitations, in others they may be a sign of a bad label assignment that can be reviewed, which is helpful.