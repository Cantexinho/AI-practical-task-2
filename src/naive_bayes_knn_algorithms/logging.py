import logging
from naive_bayes_knn_algorithms.conf.root import root_settings


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        filename=root_settings.log_path,
        level=root_settings.log_level,
        format=root_settings.log_format,
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(root_settings.log_level)

    formatter = logging.Formatter(root_settings.log_format)
    console_handler.setFormatter(formatter)

    logging.getLogger().addHandler(console_handler)
