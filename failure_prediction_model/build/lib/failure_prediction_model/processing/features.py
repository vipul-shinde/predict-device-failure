# Importing the libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from failure_prediction_model.config import config

# Creating a SMOTE based sampler.
smote = SMOTE(sampling_strategy=config.SMOTE_SAMPLING_STRATERGY)
under = RandomUnderSampler(sampling_strategy=config.UNDER_SAMPLING_STRATERGY)

# Creating an imblearn pipeline to perform SMOTE followed by undersampling
sampling_pipeline = Pipeline([
    ("smote", smote),
    ("under_sampling", under)
])

# Building a columntransformer pipeline for data preprocessing
preprocessing_pipeline = ColumnTransformer([
    ("std_scaler", StandardScaler(), config.STANDARD_SCALER_FEATURES),
    ("min_max_scaler", MinMaxScaler(), config.MIN_MAX__SCALER_FEATURES)
])
