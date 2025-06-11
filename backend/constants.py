import os

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Paths
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

BASE_DATA_DIR = os.path.join(THIS_DIR, "..", "data")
OUT_DATA_DIR = os.path.join(THIS_DIR, "..", "data-out")
BASE_MODELS_DIR = os.path.join(THIS_DIR, "models")
BASE_RESULTS_DIR = os.path.join(THIS_DIR, "..", "results")
TEST_DIR = os.path.join(THIS_DIR, "..", "tests")
FRONTEND_DIR = os.path.join(THIS_DIR, "..", "frontend")

# Data
TRAIN_ANNOTATIONS = "train-annotations.tsv"
VALID_ANNOTATIONS = "valid-annotations.tsv"
FULL_ANNOTATIONS = "full-annotations.tsv"
ANNOTATIONS_DIR = os.path.join(BASE_DATA_DIR, "annotations")
TRAIN_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, TRAIN_ANNOTATIONS)
VALID_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, VALID_ANNOTATIONS)
FULL_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, FULL_ANNOTATIONS)
DB_DATA_DIR = os.path.join(BASE_DATA_DIR, "database-format")
DB_DIR = os.path.join(THIS_DIR, "..", "db")
DB_PATH = os.path.join(DB_DIR, "sharkMatcher.db")
EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, "embeddings")
TRAINED_MODELS_DIR = os.path.join(BASE_MODELS_DIR, "trained")
IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")
SEGMENTED_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "segmented-images")
POPPED_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "popped-images")
LINEAR_PROBING_DATA_DIR = os.path.join(BASE_DATA_DIR, "linear-probing")
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw-data")
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
FRONTEND_IMAGES_DIR = os.path.join(FRONTEND_DIR, "static", "images")

SECRETS_DIR = os.path.join(THIS_DIR)

# # NEW


# NEW_ANNOTATIONS_DIR = os.path.join(BASE_DATA_DIR, "NEWannotations")
# NEW_TRAIN_ANNOTATIONS_PATH = os.path.join(NEW_ANNOTATIONS_DIR, TRAIN_ANNOTATIONS)
# NEW_VALID_ANNOTATIONS_PATH = os.path.join(NEW_ANNOTATIONS_DIR, VALID_ANNOTATIONS)
# NEW_FULL_ANNOTATIONS_PATH = os.path.join(NEW_ANNOTATIONS_DIR, FULL_ANNOTATIONS)
# NEW_RAW_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "images")
# NEW_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")

# NEW_DB_DATA_DIR = os.path.join(BASE_DATA_DIR, "NEWdatabase-format")
# NEW_DB_DIR = os.path.join(THIS_DIR, "..", "NEWdb")
# NEW_DB_PATH = os.path.join(NEW_DB_DIR, "sharkMatcher.db")

# NEW_EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, "NEWembeddings")



# Test

# Return values
SUCCESS = 0
ERROR = -1

# Models
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

# Learning
COMPUTE_PERIOD = 1
COMPUTE_PERIOD_MIN = 0
MINI_BATCH_SIZE = 16
