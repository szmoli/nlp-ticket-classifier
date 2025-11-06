import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from gensim.models import FastText
import json

import db
import preprocess
import vector
import cache

MODEL_DIR = "cache"
MODEL_SUFFIX = "from_db"
FT_MODEL_PATH = os.path.join(MODEL_DIR, f"fasttext_{MODEL_SUFFIX}.model")
CLF_PATH = os.path.join(MODEL_DIR, f"classifier_{MODEL_SUFFIX}.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, f"metrics_{MODEL_SUFFIX}.json")

def load_texts_and_labels():
    db.initialize()
    rows = db.get_all_tickets()  
    texts = []
    labels = []
    for r in rows:
        team = r["team"]
        body = r["body"] or ""
        if not team or team.strip() == "":
            continue
        texts.append(body)
        labels.append(team)
    return texts, labels

def train_fasttext(clean_docs, vector_size=150, window=7, min_count=2, epochs=50):
    from gensim.models import FastText
    model = FastText(vector_size=vector_size, window=window, min_count=min_count, sg=1)
    model.build_vocab(clean_docs)
    model.train(clean_docs, total_examples=len(clean_docs), epochs=epochs)
    return model

def train_classifier(X, y):
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    calib = CalibratedClassifierCV(estimator=clf, cv=3)
    calib.fit(X, y)
    return calib

def save_models(ft_model, clf, metrics, ft_path=FT_MODEL_PATH, clf_path=CLF_PATH, metrics_path=METRICS_PATH):
    os.makedirs(os.path.dirname(ft_path), exist_ok=True)
    ft_model.save(ft_path)
    with open(clf_path, "wb") as fh:
        pickle.dump(clf, fh)
    with open(metrics_path, "w") as mh:
        json.dump(metrics, mh, indent=2)
    print(f"[INFO] Saved FastText to {ft_path}")
    print(f"[INFO] Saved classifier to {clf_path}")
    print(f"[INFO] Saved metrics to {metrics_path}")

def load_models():
    if not os.path.exists(FT_MODEL_PATH):
        raise FileNotFoundError(f"FastText model not found: {FT_MODEL_PATH}")
    if not os.path.exists(CLF_PATH):
        raise FileNotFoundError(f"Classifier not found: {CLF_PATH}")
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Metrics not found: {METRICS_PATH}")

    ft = FastText.load(FT_MODEL_PATH)
    with open(CLF_PATH, "rb") as fh:
        clf = pickle.load(fh)
    with open(METRICS_PATH, "r") as mh:
        metrics = json.load(mh)
    return ft, clf, metrics

def info(ft_model, clf, metrics):
    return {
        "fasttext_params": {
            "vector_size": ft_model.vector_size,
            "window": ft_model.window,
            "min_count": ft_model.min_count,
            "epochs": ft_model.epochs
        },
        "classifier_params": {
            "n_estimators": clf.estimator.n_estimators,
            "class_weight": clf.estimator.class_weight,
            "cv": clf.cv
        },
        "feature_stats": {
            "vector_dim": ft_model.vector_size,
            "vocab_size": len(ft_model.wv),
        },
        "classes": clf.classes_.tolist(),
        "metrics": metrics
    }

def team(text, ft_model, clf, threshold=0.4):
    lemmas = preprocess.this(text)
    vec = vector.of(lemmas, ft_model)
    X = np.array([vec])  
    probs = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    probs_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    top_idx = int(np.argmax(probs))
    top_cls = classes[top_idx]
    top_prob = float(probs[top_idx])

    if top_prob >= threshold:
        return top_cls, top_prob, probs_dict
    return None, top_prob, probs_dict

def main():
    print("[INFO] Loading texts and labels from DB...")
    texts, labels = load_texts_and_labels()
    print(f"[INFO] {len(texts)} labeled examples loaded.")

    print("[INFO] Preprocessing texts...")
    clean_docs = [preprocess.this(text) for text in texts]

    print("[INFO] Training FastText...")
    ft_model = train_fasttext(clean_docs)

    print("[INFO] Creating document vectors...")
    X = [vector.of(doc, ft_model) for doc in clean_docs]
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

    print("[INFO] Training classifier...")
    clf = train_classifier(X_train, y_train)

    print("[INFO] Computing model metrics and confusion matrix")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    save_models(ft_model, clf, metrics)
    print("[INFO] Training finished.")

if __name__ == "__main__":
    main()
