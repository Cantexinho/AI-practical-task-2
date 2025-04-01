from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from naive_bayes_knn_algorithms.preprocessor import get_preprocessor
from naive_bayes_knn_algorithms.training.model_trainer import get_model_trainer
from naive_bayes_knn_algorithms.evaluation.evaluator import get_evaluator
from naive_bayes_knn_algorithms.logging import setup_logging

setup_logging()


def main():
    preprocessor = get_preprocessor()
    trainer = get_model_trainer()
    evaluator = get_evaluator()

    model = MultinomialNB()

    preprocessor.load_data("src/naive_bayes_knn_algorithms/data/spam.csv")
    train_test_data = preprocessor.preprocess_data()

    trainer.load_data(
        model=model,
        model_name="multinomial_spam_detection_model_001",
        text_train=train_test_data.text_train,
        label_train=train_test_data.label_train,
    )
    trained_model = trainer.train_model()
    label_predictions = trained_model.predict(train_test_data.text_test)

    evaluator.load_data(
        text_test=train_test_data.text_test,
        label_test=train_test_data.label_test,
        label_pred=label_predictions,
    )
    accuracy = evaluator.get_accuracy()
    precision = evaluator.get_precision()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")


if __name__ == "__main__":
    main()
