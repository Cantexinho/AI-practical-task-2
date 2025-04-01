from pydantic_settings import BaseSettings, SettingsConfigDict


class RootSettings(BaseSettings):

    ml_models_path: str = "src/naive_bayes_knn_algorithms/ml_models/"

    model_config = SettingsConfigDict(
        env_prefix="NBKNN_",
        env_file=".env",
        extra="ignore",
    )

    def get_ml_models_save_path(self, model_name: str) -> str:
        return f"{self.ml_models_path}/{model_name}.pkl"


settings = RootSettings()
