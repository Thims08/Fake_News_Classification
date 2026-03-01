from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_model(data, y, model_type=""):
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=42
    )

    if model_type == "SVM":
        model = LinearSVC(C=1.0, random_state=42, max_iter=10000)
    elif model_type == "logistic":
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test