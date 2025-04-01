from pydantic import BaseModel, ConfigDict

from scipy.sparse import csr_matrix
import pandas as pd


class TrainTestData(BaseModel):
    """
    Class to hold the train and test data.
    """

    text_train: csr_matrix
    text_test: csr_matrix
    label_train: pd.Series
    label_test: pd.Series

    model_config = ConfigDict(arbitrary_types_allowed=True)
