import logging

from scipy.sparse import csr_matrix
import pandas as pd

_logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self):
        self.text_test = None
        self.label_test = None
        self.label_pred = None

    def load_data(
        self,
        text_test: csr_matrix,
        label_test: pd.Series,
        label_pred: pd.Series,
    ) -> None:
        """Load the test data into the evaluator."""
        _logger.info("Loading test data into evaluator.")

        self.text_test = text_test
        self.label_test = label_test
        self.label_pred = label_pred

    def get_accuracy(self) -> float:
        """Calculate accuracy of the model."""
        _logger.info("Calculating accuracy.")

        return (self.label_test == self.label_pred).mean()

    def get_precision(self) -> float:
        """Calculate precision of the model."""
        _logger.info("Calculating precision.")

        true_positive = (
            (self.label_test == "spam") & (self.label_pred == "spam")
        ).sum()
        false_positive = (
            (self.label_test != "spam") & (self.label_pred == "spam")
        ).sum()
        return (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0
        )


_evaluator = Evaluator()


def get_evaluator() -> Evaluator:
    """Get an instance of the Evaluator."""
    return _evaluator
