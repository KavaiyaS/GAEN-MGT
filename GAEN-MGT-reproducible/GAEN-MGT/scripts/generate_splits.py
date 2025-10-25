import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

DATA_PATH = Path("data/hinglish_sentiment.csv")
assert DATA_PATH.exists(), "Place hinglish_sentiment.csv in ./data/"
df = pd.read_csv(DATA_PATH)
assert {'text','label'}.issubset(df.columns), "CSV must have 'text' and 'label'."

train, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train, val  = train_test_split(train, test_size=0.1111, stratify=train['label'], random_state=42)

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
for i,(tr,va) in enumerate(skf.split(train['text'], train['label'])):
    train.iloc[tr].to_csv(f"data/folds/fold_{i+1}_train.csv", index=False)
    train.iloc[va].to_csv(f"data/folds/fold_{i+1}_val.csv", index=False)
print("Saved splits and 5-fold files in data/folds/")
