import joblib


class ModelRunner:
    def __init__(self, model_path: str, vectorizer_path: str):
        self._model_path = model_path
        self._vectorizer_path = vectorizer_path
        self._model = self._load_model()
        self._vectorizer = self._load_vectorizer()

    def _load_model(self):
        """Load the model and vectorizer from the specified paths."""
        try:
            return joblib.load(self._model_path)
        except FileNotFoundError as e:
            raise ValueError(f"File not found: {e.filename}")

    def _load_vectorizer(self):
        """Load the vectorizer from the specified path."""
        try:
            return joblib.load(self._vectorizer_path)
        except FileNotFoundError as e:
            raise ValueError(f"File not found: {e.filename}")

    def predict(self, text: str):
        """Predict the label for the given text."""
        if self._model is None or self._vectorizer is None:
            raise ValueError(
                "Model or vectorizer not loaded. Call load_model() and load_vectorizer() first."
            )

        vectorized_text = self._vectorizer.transform([text])
        predicted_label = self._model.predict(vectorized_text)
        return predicted_label


_model_runner = ModelRunner(
    model_path="src/naive_bayes_knn_algorithms/trained_models/multinomial_spam_detection_model_001.pkl",
    vectorizer_path="src/naive_bayes_knn_algorithms/trained_models/multinomial_spam_detection_model_001_vectorizer.pkl",
)

prediction = _model_runner.predict("URGENT! You have won a 1 week FREE membership!")
print(prediction)
