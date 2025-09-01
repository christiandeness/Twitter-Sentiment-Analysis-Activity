import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC

from constants import DATASET, MODELS_DIR

MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATASET)
df.columns = ["target", "text"]

df = df[df["target"] != 2]
df["target"] = df["target"].map({0: 0, 4: 1})
df = df.reset_index(drop=True)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Class distribution:\n{df['target'].value_counts()}")

# Preprocess the Text
def _lowercase(text:str) -> str:
    return text.lower()

df["text"] = df["text"].apply(_lowercase)

#Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")


#Perform TF-IDF Vectorization
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)
joblib.dump(vec, MODELS_DIR / "tfidf_vectorizer.pkl")


#Training Models
#1. Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_vec, y_train)
y_bnb_pred = bnb.predict(X_test_vec)
print(f"BNB Accuracy: {accuracy_score(y_test, y_bnb_pred)}")
print(f"BNB Classification Report:\n{classification_report(y_test, y_bnb_pred)}")
joblib.dump(bnb, MODELS_DIR / "bnb_model.pkl")

#2. SVC (Support Vector Machine Classifier)
lsvc = LinearSVC()
lsvc.fit(X_train_vec, y_train)
y_lsvc_pred = lsvc.predict(X_test_vec)
print(f"SVC Accuracy: {accuracy_score(y_test, y_lsvc_pred)}")
print(f"SVC Classification Report:\n{classification_report(y_test, y_lsvc_pred)}")
joblib.dump(lsvc, MODELS_DIR / "lsvc_model.pkl")

#3. Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_vec, y_train)
y_lr_pred = lr.predict(X_test_vec)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_lr_pred)}")
print(f"Logistic Regression Classification Report:\n{classification_report(y_test, y_lr_pred)}")
joblib.dump(lr, MODELS_DIR / "lr_model.pkl")

# --- Inference Demo ---
print("\n=== Inference Demo ===")

# Load vectorizer and models
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
bnb = joblib.load(MODELS_DIR / "bnb_model.pkl")
svc = joblib.load(MODELS_DIR / "lsvc_model.pkl")
logreg = joblib.load(MODELS_DIR / "lr_model.pkl")

# Sample tweets to test
samples = [
    "That fit is so fire!", 
    "OMG, Fifth Harmony is back!.", 
    "Jump by Blackpink is not that good. I said what I said.",
    "Twice's comeback is not bad, but could have been much better.",
    "Oh my God, this song is so lit!",
    "Everything's Gnarly!"
]

# Transform with the vectorizer
X_samples = vectorizer.transform(samples)

# Predict with each model
for model_name, model in [("BNB", bnb), ("SVC", svc), ("LogReg", logreg)]:
    preds = model.predict(X_samples)
    print(f"\n{model_name} predictions:")
    for text, pred in zip(samples, preds):
        label = "Positive" if pred == 1 else "Negative"
        print(f"  \"{text}\" -> {label}")
