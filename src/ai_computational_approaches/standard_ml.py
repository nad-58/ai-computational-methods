"""Standard machine learning algorithms using scikit-learn."""
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def decision_tree_demo(random_state: int = 7) -> dict:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
    model = DecisionTreeClassifier(max_depth=3, random_state=random_state).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"method": "decision_tree", "accuracy": float(accuracy_score(y_test, pred)), "depth": model.get_depth()}


def random_forest_demo(random_state: int = 7) -> dict:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, oob_score=True).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "method": "random_forest",
        "accuracy": float(accuracy_score(y_test, pred)),
        "oob_score": float(model.oob_score_),
        "feature_importance": model.feature_importances_.round(3).tolist(),
    }


def linear_regression_demo(random_state: int = 7) -> dict:
    X, y = make_regression(n_samples=200, n_features=4, noise=15.0, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"method": "linear_regression", "mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred))}


def logistic_regression_demo(random_state: int = 7) -> dict:
    X, y = make_classification(n_samples=300, n_features=8, n_informative=4, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"method": "logistic_regression", "accuracy": float(accuracy_score(y_test, pred)), "f1": float(f1_score(y_test, pred))}


def knn_demo(random_state: int = 7) -> dict:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
    model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"method": "knn", "accuracy": float(accuracy_score(y_test, pred)), "k": 5}


def naive_bayes_demo(random_state: int = 7) -> dict:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, stratify=y)
    model = GaussianNB().fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"method": "naive_bayes", "accuracy": float(accuracy_score(y_test, pred))}


def run_all_standard_ml() -> list[dict]:
    return [
        decision_tree_demo(),
        random_forest_demo(),
        linear_regression_demo(),
        logistic_regression_demo(),
        knn_demo(),
        naive_bayes_demo(),
    ]


if __name__ == "__main__":
    for result in run_all_standard_ml():
        print(result)
