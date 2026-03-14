"""
Spam Classifier — Apache SpamAssassin Public Corpus

Pipeline:
  1. Download & extract spam/ham email archives
  2. Parse raw emails: strip headers, extract text from HTML with BeautifulSoup,
     apply regex transforms (URLs → URL, numbers → NUMBER)
  3. Custom sklearn transformer → CountVectorizer (word-presence or word-count)
  4. Compare Logistic Regression, Naive Bayes, and LinearSVC
  5. Report precision & recall for each
"""

import email
import re
import tarfile
import urllib.request
from pathlib import Path

import joblib
import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "spam_assassin"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"


def get_model_path(accuracy: float) -> Path:
    return MODELS_DIR / f"model_task4_{accuracy * 100:.0f}pct.joblib"


def find_latest_model() -> Path:
    matches = sorted(MODELS_DIR.glob("model_task4_*pct.joblib"))
    if not matches:
        raise FileNotFoundError("No task-4 model found. Run main() first.")
    return matches[-1]

DATASETS = {
    "spam": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
}

# ---------------------------------------------------------------------------
# 1. Download & extract
# ---------------------------------------------------------------------------

def download_and_extract():
    for label, url in DATASETS.items():
        dest = DATA_DIR / label
        if dest.exists() and any(dest.iterdir()):
            print(f"[{label}] already extracted, skipping.")
            continue
        archive = DATA_DIR / url.split("/")[-1]
        if not archive.exists():
            print(f"Downloading {label} from {url} ...")
            urllib.request.urlretrieve(url, archive)
        print(f"Extracting {archive.name} ...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(DATA_DIR / label)
        print(f"[{label}] done.")


def load_emails(label: str) -> list[bytes]:
    """Return raw bytes for every file under DATA_DIR/<label>."""
    root = DATA_DIR / label
    files = [f for f in root.rglob("*") if f.is_file() and f.name != "cmds"]
    emails = []
    for f in files:
        try:
            emails.append(f.read_bytes())
        except Exception:
            pass
    return emails


# ---------------------------------------------------------------------------
# 2. Email preprocessing transformer
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """Strip HTML tags and return plain text using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
PUNCT_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

_stemmer = PorterStemmer()


def parse_email(raw: bytes) -> str:
    """Extract body text from a raw email (bytes)."""
    msg = email.message_from_bytes(raw)
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                parts.append(part.get_payload(decode=True).decode("utf-8", errors="ignore"))
            elif ct == "text/html":
                html = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                parts.append(_html_to_text(html))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode("utf-8", errors="ignore")
            if msg.get_content_type() == "text/html":
                text = _html_to_text(text)
            parts.append(text)
    return " ".join(parts)


class EmailPreprocessor(BaseEstimator, TransformerMixin):
    """
    Converts raw email bytes → cleaned string.

    Hyperparameters
    ---------------
    strip_headers   : bool  — ignore email header lines
    to_lowercase    : bool  — lowercase the entire body
    replace_urls    : bool  — swap every URL with the token 'URL'
    replace_numbers : bool  — swap every number with the token 'NUMBER'
    remove_punct    : bool  — strip punctuation
    stem            : bool  — apply PorterStemmer to each token
    """

    def __init__(
        self,
        strip_headers: bool = True,
        to_lowercase: bool = True,
        replace_urls: bool = True,
        replace_numbers: bool = True,
        remove_punct: bool = True,
        stem: bool = False,
    ):
        self.strip_headers = strip_headers
        self.to_lowercase = to_lowercase
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.remove_punct = remove_punct
        self.stem = stem

    # sklearn requires fit to return self
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self._process(raw) for raw in X]

    def _process(self, raw: bytes) -> str:
        text = parse_email(raw)

        if self.to_lowercase:
            text = text.lower()
        if self.replace_urls:
            text = URL_RE.sub(" URL ", text)
        if self.replace_numbers:
            text = NUMBER_RE.sub(" NUMBER ", text)
        if self.remove_punct:
            text = PUNCT_RE.sub(" ", text)

        text = MULTI_SPACE_RE.sub(" ", text).strip()

        if self.stem:
            text = " ".join(_stemmer.stem(w) for w in text.split())

        return text


# ---------------------------------------------------------------------------
# 3. Build dataset
# ---------------------------------------------------------------------------

def build_dataset():
    nltk.download("punkt", quiet=True)
    download_and_extract()

    spam_raw = load_emails("spam")
    ham_raw = load_emails("ham")

    print(f"Spam emails : {len(spam_raw)}")
    print(f"Ham  emails : {len(ham_raw)}")

    X = spam_raw + ham_raw
    y = np.array([1] * len(spam_raw) + [0] * len(ham_raw))
    return X, y


# ---------------------------------------------------------------------------
# 4. Train & evaluate
# ---------------------------------------------------------------------------

def make_pipeline(classifier):
    return Pipeline([
        ("prep",  EmailPreprocessor(stem=False)),
        ("vect",  CountVectorizer(binary=True, max_features=10_000)),
        ("clf",   classifier),
    ])


def evaluate(name, pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    return accuracy, pipeline


def main():
    X, y = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "LinearSVC": LinearSVC(max_iter=2000, C=1.0, random_state=42),
    }

    best_acc, best_pipeline, best_name = 0.0, None, ""
    for name, clf in classifiers.items():
        acc, pipe = evaluate(name, make_pipeline(clf), X_train, y_train, X_test, y_test)
        if acc > best_acc:
            best_acc, best_pipeline, best_name = acc, pipe, name

    # Cross-val on the best typical performer (LR) for a more robust estimate
    print("\n--- 3-fold CV (Logistic Regression, scoring=f1) ---")
    lr_pipe = make_pipeline(LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    cv_scores = cross_val_score(lr_pipe, X_train, y_train, cv=3, scoring="f1", n_jobs=-1)
    print(f"F1 scores : {cv_scores.round(4)}")
    print(f"Mean F1   : {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")

    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(best_acc)
    joblib.dump({"pipeline": best_pipeline, "name": best_name, "accuracy": best_acc}, model_path)
    print(f"\nBest model ({best_name}, {best_acc:.2%}) saved to: {model_path}")


if __name__ == "__main__":
    main()
