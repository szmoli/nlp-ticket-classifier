import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from gensim.models import FastText

import db
import preprocess
import vector
import cache

MODEL_DIR = "cache"
MODEL_PFX = "from_db"
FT_MODEL_PATH = os.path.join(MODEL_DIR, f"fasttext_{MODEL_PFX}.model")
CLF_PATH = os.path.join(MODEL_DIR, f"classifier_{MODEL_PFX}.pkl")

def load_texts_and_labels():
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
    # calib = CalibratedClassifierCV(base_estimator=clf, cv=3)
    calib.fit(X, y)
    return calib

def save_models(ft_model, clf, ft_path=FT_MODEL_PATH, clf_path=CLF_PATH):
    os.makedirs(os.path.dirname(ft_path), exist_ok=True)
    ft_model.save(ft_path)
    with open(clf_path, "wb") as fh:
        pickle.dump(clf, fh)
    print(f"[INFO] Saved FastText -> {ft_path}")
    print(f"[INFO] Saved classifier -> {clf_path}")

def load_models():
    if not os.path.exists(FT_MODEL_PATH):
        raise FileNotFoundError(f"FastText model not found: {FT_MODEL_PATH}")
    if not os.path.exists(CLF_PATH):
        raise FileNotFoundError(f"Classifier not found: {CLF_PATH}")

    ft = FastText.load(FT_MODEL_PATH)
    with open(CLF_PATH, "rb") as fh:
        clf = pickle.load(fh)
    return ft, clf

def team(text, ft_model, clf, threshold=0.7):
    lemmas = preprocess.this(text)
    vec = vector.of(lemmas, ft_model)
    X = np.array([vec])  # 2D shape expected by sklearn

    # Prefer predict_proba
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
    else:
        # fallback to decision_function -> softmax
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)[0]
            ex = np.exp(scores - np.max(scores))
            probs = ex / ex.sum()
        else:
            # as last resort use predict and return full confidence
            pred = clf.predict(X)[0]
            return pred, 1.0, {pred: 1.0}

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

    if len(texts) < 10:
        raise RuntimeError("Not enough labeled examples to train a classifier.")

    print("[INFO] Preprocessing texts...")
    clean_docs = [preprocess.this(text) for text in texts]

    print("[INFO] Training FastText...")
    ft_model = train_fasttext(clean_docs)

    print("[INFO] Creating document vectors...")
    X = [vector.of(doc, ft_model) for doc in clean_docs]

    print("[INFO] Training classifier (with probability calibration)...")
    clf = train_classifier(X, labels)

    save_models(ft_model, clf)
    print("[INFO] Training finished.")

if __name__ == "__main__":
    main()
