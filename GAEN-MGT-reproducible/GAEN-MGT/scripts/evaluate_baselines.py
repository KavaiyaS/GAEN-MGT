import re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def clean(t): return re.sub(r"[^A-Za-z0-9\s]", " ", str(t)).lower()

df_tr = pd.read_csv("data/train.csv")
df_va = pd.read_csv("data/val.csv")
df_te = pd.read_csv("data/test.csv")
df = pd.concat([df_tr, df_va], ignore_index=True)

Xtr, ytr = df["text"].astype(str).apply(clean), df["label"]
Xte, yte = df_te["text"].astype(str).apply(clean), df_te["label"]

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True)
Xtr_t = tfidf.fit_transform(Xtr)
Xte_t = tfidf.transform(Xte)

svm = LinearSVC(random_state=42).fit(Xtr_t, ytr)
pred = svm.predict(Xte_t)
acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average="macro")
print(f"LinearSVM — Acc: {acc:.3f} | Macro-F1: {f1:.3f}")
Path("results/metrics").mkdir(parents=True, exist_ok=True)
(Path("results/metrics")/"baseline_linear_svm.txt").write_text(classification_report(yte, pred, digits=3))
