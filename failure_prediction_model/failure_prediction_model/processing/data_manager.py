# Importing the libraries
from ast import Str
import pandas as pd
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline

from failure_prediction_model import __version__ as _version
from failure_prediction_model.config import config


def load_dataset(*, file_name: str, save: bool = False, save_as: str = None, train: bool = True) -> pd.DataFrame:
    """Load, modify and save a training or test dataset.

    Parameters
    ----------
    file_name : str
        name of the input data file
    save : bool, optional
        True if you want to save the modified dataset, by default False
    save_as : str, optional
        name to save the file as, by default None
    train : bool, optional
        False if you are passing a test dataset, by default True

    Returns
    -------
    pd.DataFrame
        _description_
    """

    use_columns = None

    if train:
        use_columns = config.TRAIN_FEATURES
    else:
        use_columns = config.TEST_FEATURES

    df = pd.read_csv(
        Path(f"{config.RAW_DATA_DIR}/{file_name}"), usecols=use_columns)

    if save:
        df.to_csv(f"{config.INTERIM_DATA_DIR}/{save_as}", index=False)

    return df


def load_interim_data(*, file_name: str, train: bool = True) -> pd.DataFrame:

    use_columns = None

    if train:
        use_columns = config.TRAIN_FEATURES
    else:
        use_columns = config.TEST_FEATURES

    dataframe = pd.read_csv(
        Path(f"{config.INTERIM_DATA_DIR}/{file_name}"), usecols=use_columns)

    return dataframe


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Save the versioned pipeline and overwrite any previous saved models

    Parameters
    ----------
    pipeline_to_persist : Pipeline
        the sklearn pipeline to be saved
    """

    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def remove_old_pipelines(*, files_to_keep: str) -> None:
    """
    Remove old model pipelines.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a saved pipeline from a pickle file

    Parameters
    ----------
    file_name : str
        name of the sklearn pipeline file

    Returns
    -------
    Pipeline
        pipeline
    """

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_pipeline = joblib.load(filename=file_path)

    return trained_pipeline
