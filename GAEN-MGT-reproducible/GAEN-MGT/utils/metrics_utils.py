from sklearn.metrics import accuracy_score, f1_score, classification_report
def summarize(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(y_true, y_pred, digits=3)
    }
