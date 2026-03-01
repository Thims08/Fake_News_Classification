import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(classification_report(
        y_test, y_pred,
        target_names=['fake', 'real']
    ))

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


def plot_confusion_matrix(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Fake', 'Real']
    )
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()


def plot_roc(model, X_test, y_test, title):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.show()


def plot_pr_curve(model, X_test, y_test, title):
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.show()