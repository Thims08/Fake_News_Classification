from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def tune_svm(X, y):
    svm_params = {
        "C": [0.1, 1, 5, 10],
        "class_weight": ["balanced", None],
        "max_iter": [5000, 10000, 20000]
    }

    svm = LinearSVC(random_state=42)

    grid = GridSearchCV(
        svm,
        svm_params,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X, y)

    print("Best SVM Params:", grid.best_params_)
    return grid.best_estimator_


def tune_logistic(X, y):
    log_params = {
        "C": [0.1, 1.0, 5, 10],
        "penalty": ["l2"],
        "class_weight": ["balanced", None],
        "solver": ["lbfgs", "liblinear"]
    }

    lr = LogisticRegression(max_iter=2000)

    grid = GridSearchCV(
        lr,
        log_params,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X, y)

    print("Best Logistic Params:", grid.best_params_)
    return grid.best_estimator_