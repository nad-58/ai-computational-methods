"""Transfer-learning style example using reusable feature transformations."""
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def transfer_learning_demo(random_state: int = 7) -> dict:
    """Demonstrate feature-representation reuse with a synthetic target task.

    This is not a large pre-trained model. It is a lightweight educational analogy:
    learn a reusable representation, then train a smaller target classifier.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        class_sep=1.2,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)

    representation = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=8, random_state=random_state)),
    ])
    X_train_rep = representation.fit_transform(X_train)
    X_test_rep = representation.transform(X_test)

    classifier = LogisticRegression(max_iter=1000).fit(X_train_rep, y_train)
    pred = classifier.predict(X_test_rep)
    return {
        "method": "transfer_learning_style_feature_reuse",
        "representation": "standard_scaler + PCA",
        "target_classifier": "logistic_regression",
        "accuracy": float(accuracy_score(y_test, pred)),
    }


if __name__ == "__main__":
    print(transfer_learning_demo())
