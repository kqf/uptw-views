from model.model import build_model, read_dataset
from sklearn.metrics import classification_report


def main():
    X_tr, X_te, y_tr, y_te = read_dataset()
    model = build_model()
    model.fit(X_tr, y_tr)
    print("Train train score", model.score(X_tr, y_tr))
    print("Train test score ", model.score(X_te, y_te))
    print("Train set")
    print(classification_report(model.predict(X_tr), y_tr))
    print("Test set")
    print(classification_report(model.predict(X_te), y_te))
    build_model()


if __name__ == "__main__":
    main()
