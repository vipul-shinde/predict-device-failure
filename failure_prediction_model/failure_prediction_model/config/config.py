# Importing the libraries
from pathlib import Path
import failure_prediction_model

# Project Directories
PACKAGE_ROOT = Path(failure_prediction_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "data"
RAW_DATA_DIR = DATASET_DIR / "raw"
INTERIM_DATA_DIR = DATASET_DIR / "interim"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"

# Training dataset path
TRAIN_DATA = "predict_failure.csv"
INTERIM_TRAIN_DATA = "train.csv"
INTERIM_TEST_DATA = "test.csv"

# Input parameters for data preprocessing
TARGET = "failure"
TEST_SIZE = 0.2
SEED = 42
N_SPLITS = 10

# Input parameters for data samplking techniques
SMOTE_SAMPLING_STRATERGY = 0.1
UNDER_SAMPLING_STRATERGY = 0.75

# Features to be dropped
DROP_ATTRIBUTES = ["date", "device", "attribute7"]

# Train and Test features to select
TRAIN_FEATURES = ["attribute1", "attribute2", "attribute3",
                  "attribute4", "attribute5", "attribute6", "attribute8", "attribute9", "failure"]
TEST_FEATURES = ["attribute1", "attribute2", "attribute3",
                 "attribute4", "attribute5",  "attribute6", "attribute8", "attribute9"]

# List of features to be scaled based on Standardization/Normalization
STANDARD_SCALER_FEATURES = ["attribute2", "attribute3",
                            "attribute4", "attribute5", "attribute8", "attribute9"]
MIN_MAX__SCALER_FEATURES = ["attribute1", "attribute6"]

# Package name
PACKAGE_NAME = "failure_detction_model"

# Pipeline name
PIPELINE_NAME = "failure_detction_model"
PIPELINE_SAVE_FILE = "failure_detction_model_v"
