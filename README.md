# shark-matcher

## Directory Structure

### /backend
All code responsible for data processing, retrieval, and handling database operations.
- /db - All database CRUD functionality.
- /evaluation - Model and retrieval evaluation and visualization scripts. 
- /learning - Code for training models including evaluator for in training, loss function implementation, training scripts and data loaders for contrastive learning.
- /models - Code with architecture for untrained backbone and projection head, contains trained directory where trained models are stored.
- /retrieval - Instance retrieval code responsible for applying trained model to generate and search through embeddings for the data and provide retrieval results.
- utils.py - General utility functions used throughout the backend directory.
- constants.py - Constants used throughout the backend directory.
- analyze_data.py - Data analysis utilities for data cleaning and summarization.
- convert_data.py - Parses data in the database and produces annotations files for use during model training and retrieval. Also prepares image directory for use in model training and retrieval.

### /data
All data in both raw format received parsed format, also stores trained models for use by Shark Matcher
- /annotations - Parsed annotations files (tsv files storing metadata and match information on each image). These files are used during model training and retrieval.
- /database-format. CSV files containing data formatted according to the database schema for loading into the database. These 
files are produced by /db/scripts reading data in this /data directory.
- /embeddings - Pytorch (.pt) embeddings files and dataset files storing cached dataset, embeddings and nearest neighbor data.
- /images - Flat directory containing all parsed images.
- /segmented-images - Flat directory containing all segmented images. These images should correspond directly to those in the /images directory, with the same names.
- /raw-data - Unparsed matching spreadsheet csv files and unparsed images in nested directory structure as received from Block lab or other original source.

### /data_out
Directory where database snapshots, logs, and output files appear.

### /db
Database directory with database schema, files, population and snapshot scripts.
- /scripts - Scripts for database population, snapshots, and debugging.
- create_database.sql - Database schema.
- sharkMatcher.db - Sqlite database file.

## Workflow
#### Prerequisites:
- Ensure dependencies (`requirements.txt`).
    pip install -r requirements.txt

1. Clone or pull the latest version of the shark-matcher repository.
2. Assure data directory is populated with all the necessary files and structure outlined in the **data** section above.
- Critical items are a populated /images directory and either a "raw annotations" file in raw-data/annotations or a database snapshot in /database-format that can be used to populate the database to complete any of the subsequent operations.

#### Model Training:
1. Make sure that all necessary images are located in the data/raw-data/images directory and that the data/images directory is empty or free of old/irrelevant data. Placing new images in raw-data/images is needed so they can be parsed and converted by the convert_data script which copies them to the data/imges directory in the next step.
2. Format the images and populate the formatted image directory data/images by running the format_image_directory() method in convert_data.py.
3. Populate the database with data from a raw annotations file or database snapshot by running `./db/scripts/populate_db.sh [-h for info]` See the options for snapshot vs annotations population.
- `./db/scripts/print_db.sh` can print a preview of the database to verify.
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