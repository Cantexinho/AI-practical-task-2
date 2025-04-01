import pandas as pd
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix

from naive_bayes_knn_algorithms.models.train_test import TrainTestData

_logger = logging.getLogger(__name__)


class PreProcessor:
    def __init__(self, vectorizer: TfidfVectorizer = None):
        self._vectorizer = vectorizer
        self._df = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the dataset from the specified file path."""
        _logger.info("Loading data from file.")
        try:
            self._df = pd.read_csv(file_path, encoding="ISO-8859-1")
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")

        self._df = self._df.loc[:, ~self._df.columns.str.contains("^Unnamed")]
        print(self._df.head())

        return self._df

    def _split_train_test(
        self, vectorized_text: csr_matrix, labels: pd.Series
    ) -> TrainTestData:
        """Split the data into training and testing sets."""
        _logger.info("Splitting data into training and testing sets.")

        if self._df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        text_train, text_test, label_train, label_test = train_test_split(
            vectorized_text, labels, test_size=0.2, random_state=42
        )

        return TrainTestData(
            text_train=text_train,
            text_test=text_test,
            label_train=label_train,
            label_test=label_test,
        )

    def preprocess_data(self) -> TrainTestData:
        """Preprocess the data by renaming columns."""
        _logger.info("Preprocessing data.")

        if self._df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self._df.rename(columns={"v1": "label", "v2": "text"}, inplace=True)

        vectorized_text = self._vectorizer.fit_transform(self._df["text"])
        labels = self._df["label"]

        split_train_test_data = self._split_train_test(vectorized_text, labels)

        return split_train_test_data


_pre_processor = PreProcessor(vectorizer=TfidfVectorizer(stop_words="english"))


def get_preprocessor() -> PreProcessor:
    """Get an instance of the PreProcessor."""
    return _pre_processor
