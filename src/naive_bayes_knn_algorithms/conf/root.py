import logging

import time

from pydantic_settings import BaseSettings, SettingsConfigDict


class RootSettings(BaseSettings):

    ml_models_path: str
    log_format: str = "%(asctime)s\t%(levelname)s %(name)s: %(module)s.py %(message)s"
    log_level: int = logging.DEBUG

    model_config = SettingsConfigDict(
        env_prefix="NBKNN_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def log_path(self) -> str:
        return f"src/naive_bayes_knn_algorithms/logs/model_training_{str(int(time.time()))}.log"

    def get_ml_models_save_path(self, model_name: str) -> str:
        return f"{self.ml_models_path}/{model_name}.pkl"


root_settings = RootSettings()
