from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from naive_bayes_knn_algorithms.logging_setup import setup_logging
from naive_bayes_knn_algorithms.conf.root import root_settings

from naive_bayes_knn_algorithms.worker import Worker

setup_logging()


def main():
    naive_bayes_worker = Worker(
        model=MultinomialNB(),
        data_location=root_settings.dataset_path,
        model_name="naive_bayes_spam_detection_model_001",
    )
    knn_worker = Worker(
        model=KNeighborsClassifier(),
        data_location=root_settings.dataset_path,
        model_name="multinomial_spam_detection_model_001",
    )

    naive_bayes_worker.run()
    knn_worker.run()


if __name__ == "__main__":
    main()
