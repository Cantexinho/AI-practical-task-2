import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from scipy.sparse import csr_matrix
import pandas as pd

from naive_bayes_knn_algorithms.conf.root import settings


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.text_train = None
        self.label_train = None

    def _save_model(self) -> None:
        """Save the trained model to a file."""

        joblib.dump(
            self.model,
            settings.get_ml_models_save_path(self.model_name),
        )

    def load_data(
        self,
        model: MultinomialNB | KNeighborsClassifier,
        model_name: str,
        text_train: csr_matrix,
        label_train: pd.Series,
    ) -> None:
        """Load the training data into the trainer."""
        self.model = model
        self.model_name = model_name
        self.text_train = text_train
        self.label_train = label_train

    def train_model(self) -> MultinomialNB | KNeighborsClassifier:
        """Train the model with the provided training data."""

        self.model.fit(self.text_train, self.label_train)
        self._save_model()

        return self.model
