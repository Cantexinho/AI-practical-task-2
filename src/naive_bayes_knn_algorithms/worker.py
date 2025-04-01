from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from naive_bayes_knn_algorithms.preprocessor import get_preprocessor
from naive_bayes_knn_algorithms.training.model_trainer import get_model_trainer
from naive_bayes_knn_algorithms.evaluation.evaluator import get_evaluator


class Worker:
    def __init__(
        self,
        model: MultinomialNB | KNeighborsClassifier,
        data_location: str,
        model_name: str,
    ):
        self.preprocessor = get_preprocessor()
        self.trainer = get_model_trainer()
        self.evaluator = get_evaluator()
        self.model = model
        self.data_location = data_location
        self.model_name = model_name

    def run(self):
        """Run the worker."""
        self.preprocessor.load_data(self.data_location)
        train_test_data = self.preprocessor.preprocess_data()

        self.trainer.load_data(
            model=self.model,
            model_name=self.model_name,
            text_train=train_test_data.text_train,
            label_train=train_test_data.label_train,
        )
        trained_model = self.trainer.train_model()
        label_predictions = trained_model.predict(train_test_data.text_test)

        self.evaluator.load_data(
            text_test=train_test_data.text_test,
            label_test=train_test_data.label_test,
            label_pred=label_predictions,
        )
        accuracy = self.evaluator.get_accuracy()
        precision = self.evaluator.get_precision()

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
