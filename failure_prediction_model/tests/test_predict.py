# Importing the libraries
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from failure_prediction_model.pipeline import vc_pipeline
from failure_prediction_model.processing.features import sampling_pipeline
from failure_prediction_model.predict import make_predictions


def test_prediction_model(sample_data, save_as="sample_out"):
    df = sample_data

    results = make_predictions(sample_data, save_as=save_as)

    assert len(results["predictions"]) == len(df)


def test_prediction_quality(input_data):

    expected_roc_auc = 0.85

    X = input_data.drop("failure", axis=1)
    y = input_data.loc[:, "failure"]

    # Splitting the dataset into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    # Resampling the training dataset to tackle imbalance problem
    X_train_rs, y_train_rs = sampling_pipeline.fit_resample(X_train, y_train)

    vc_pipeline.fit(X_train_rs, y_train_rs)

    y_pred = vc_pipeline.predict_proba(X_test)
    roc_auc_score = roc_auc_score(y_test, y_pred[:, 1])

    assert y_pred is not None
    assert roc_auc_score > expected_roc_auc
