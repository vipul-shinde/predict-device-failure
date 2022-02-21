# Importing the libraries
import logging

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from config import config
from processing.features import preprocessing_pipeline

logger = logging.getLogger(__name__)

# Creating a Voting Classifier consisting of our top 3 performing models i.e GB, LR and KNN Classifiers
vc_pipeline = Pipeline([
    ("preprocess", preprocessing_pipeline),
    ("v_clf", VotingClassifier(estimators=[("gb", GradientBoostingClassifier(n_estimators=500, learning_rate=0.01)),
                                           ("knn", KNeighborsClassifier(
                                               n_neighbors=25)),
                                           ("lr", LogisticRegression(solver="saga", max_iter=2000, class_weight="balanced"))],
                               voting="soft"))
])
