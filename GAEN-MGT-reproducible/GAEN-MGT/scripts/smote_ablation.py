import re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from pathlib import Path
import matplotlib.pyplot as plt

def clean(t): return re.sub(r"[^A-Za-z0-9\s]", " ", str(t)).lower()

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
Xtr, ytr = train["text"].astype(str).apply(clean), train["label"]
Xte, yte = test["text"].astype(str).apply(clean), test["label"]

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), sublinear_tf=True)
Xtr_t = tfidf.fit_transform(Xtr); Xte_t = tfidf.transform(Xte)

lr_ns = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_t, ytr)
p_lr_ns = lr_ns.predict(Xte_t)
acc_lr_ns = accuracy_score(yte, p_lr_ns); f1_lr_ns = f1_score(yte, p_lr_ns, average="macro")

svd = TruncatedSVD(n_components=300, random_state=42)
Xtr_svd = svd.fit_transform(Xtr_t); Xte_svd = svd.transform(Xte_t)
sm = SMOTE(random_state=42, k_neighbors=5)
Xtr_bal, ytr_bal = sm.fit_resample(Xtr_svd, ytr)
lr_sm = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_bal, ytr_bal)
p_lr_sm = lr_sm.predict(Xte_svd)
acc_lr_sm = accuracy_score(yte, p_lr_sm); f1_lr_sm = f1_score(yte, p_lr_sm, average="macro")

svm_ns = LinearSVC(random_state=42).fit(Xtr_t, ytr)
p_svm_ns = svm_ns.predict(Xte_t)
acc_svm_ns = accuracy_score(yte, p_svm_ns); f1_svm_ns = f1_score(yte, p_svm_ns, average="macro")

Xtr_bal_svm, ytr_bal_svm = sm.fit_resample(Xtr_svd, ytr)
svm_sm = LinearSVC(random_state=42).fit(Xtr_bal_svm, ytr_bal_svm)
p_svm_sm = svm_sm.predict(Xte_svd)
acc_svm_sm = accuracy_score(yte, p_svm_sm); f1_svm_sm = f1_score(yte, p_svm_sm, average="macro")

Path("results/metrics").mkdir(parents=True, exist_ok=True)
import json
(Path("results/metrics")/"smote_vs_nosmote.json").write_text(json.dumps({
    "LR_NoSMOTE":{"acc":acc_lr_ns,"f1":f1_lr_ns},
    "LR_SMOTE":{"acc":acc_lr_sm,"f1":f1_lr_sm},
    "SVM_NoSMOTE":{"acc":acc_svm_ns,"f1":f1_svm_ns},
    "SVM_SMOTE":{"acc":acc_svm_sm,"f1":f1_svm_sm}
}, indent=2))

labels = ["LR No-SMOTE","LR SMOTE+SVD","SVM No-SMOTE","SVM SMOTE+SVD"]
accs = [acc_lr_ns, acc_lr_sm, acc_svm_ns, acc_svm_sm]
f1s  = [f1_lr_ns,  f1_lr_sm,  f1_svm_ns,  f1_svm_sm]
x = np.arange(len(labels)); w = 0.38
plt.figure(figsize=(9,5))
plt.bar(x - w/2, accs, width=w, label="Accuracy")
plt.bar(x + w/2, f1s,  width=w, label="Macro-F1")
plt.legend(); plt.xticks(x, labels, rotation=10); plt.ylim(0,1); plt.ylabel("Score")
plt.title("TF-IDF Baselines — SMOTE vs No-SMOTE (LR & LinearSVM)")
Path("results/plots").mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig("results/plots/smote_vs_no_smote_comparison.png", dpi=200)
print("Saved metrics and plot to results/")
