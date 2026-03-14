# Chapter 03 — ML Exercises

Hands-on exercises covering classification, data augmentation, feature engineering, and NLP pipelines using scikit-learn.

## Setup

```bash
# Install dependencies (requires uv)
uv sync
```

## Quick start — Makefile

```bash
make help          # Show all available commands

make train1        # Train MNIST KNN classifier
make train2        # Train MNIST KNN + data augmentation
make train3        # Train Titanic logistic regression
make train4        # Train spam classifier
make train-all     # Train all four tasks in order

make test1         # Evaluate task-1 model
make test2         # Evaluate task-2 model
make test3         # Evaluate task-3 model
make test4         # Evaluate task-4 model
make test-all      # Evaluate all four tasks in order
```

Alternatively, run scripts directly from the project root with `uv run python -m src.tasks.chapter_03.task_N.<script>`.

---

## Task 1 — MNIST KNN Classifier (`>97% accuracy`)

**Goal:** Classify handwritten digits (0–9) from the MNIST dataset using K-Nearest Neighbors. Uses PCA for dimensionality reduction and GridSearchCV for hyperparameter tuning.

### Train
```bash
uv run python -m src.tasks.chapter_03.task_1.mnist_classifier
```
Trains on 60 000 samples, searches `n_neighbors ∈ {3,4,5}` and `weights ∈ {uniform, distance}`.
Saves model to: `models/model_task1_<acc>pct.joblib`

### Test
```bash
uv run python -m src.tasks.chapter_03.task_1.test_model
```
Loads the saved model, evaluates on the 10 000-sample MNIST test set, and prints accuracy + classification report + confusion matrix.

---

## Task 2 — MNIST KNN with Data Augmentation

**Goal:** Improve the Task 1 model by expanding the training set 5× via pixel-shift augmentation (left, right, up, down).

### Train
```bash
uv run python -m src.tasks.chapter_03.task_2.mnist_augmented
```
Generates 300 000 augmented samples, applies PCA, and trains KNN.
Saves model to: `models/model_task2_<acc>pct.joblib`
Also saves sample visualizations to `task_2/original_images/` and `task_2/transformed_images/`.

### Test
```bash
uv run python -m src.tasks.chapter_03.task_2.test_model
```
Same evaluation as Task 1 — accuracy, classification report, and confusion matrix on the full MNIST test set.

---

## Task 3 — Titanic Survival Classifier (Logistic Regression)

**Goal:** Predict passenger survival on the Titanic using logistic regression with manual feature engineering.

**Feature transformations applied:**

| Feature | Transformation |
|---|---|
| `sex` | Binary encoded → `is_female` |
| `age` | Median imputation + binned into 5 age groups |
| `fare` | Median imputation + `log1p` to reduce skew |
| `sibsp` + `parch` | Combined → `family_size` + `is_alone` flag |
| `embarked` | Missing filled with `S` + one-hot encoded |

### Train
```bash
uv run python -m src.tasks.chapter_03.task_3.titanic_classifier
```
Dataset is fetched automatically from OpenML.
Saves model to: `models/model_task3_<acc>pct.joblib`

### Test
```bash
uv run python -m src.tasks.chapter_03.task_3.test_model
```
Evaluates on the held-out 20% test split. Reports accuracy, precision, recall, classification report, and confusion matrix.

---

## Task 4 — Spam Classifier (Apache SpamAssassin corpus)

**Goal:** Build a spam/ham email classifier using a full NLP pipeline: email parsing, HTML extraction, regex transforms, bag-of-words vectorization, and multiple classifiers. Saves the best-performing model.

**Pipeline stages:**

1. **`EmailPreprocessor`** — custom sklearn transformer with configurable options:
   - `strip_headers` — discard email headers
   - `to_lowercase` — normalize case
   - `replace_urls` — substitute URLs with `URL` token
   - `replace_numbers` — substitute numbers with `NUMBER` token
   - `remove_punct` — strip punctuation
   - `stem` — apply NLTK PorterStemmer
   - Uses **BeautifulSoup** to extract text from `text/html` parts
   - Uses **regex** for URL, number, and punctuation transforms

2. **`CountVectorizer(binary=True, max_features=10_000)`** — word-presence bag-of-words

3. Classifier (Logistic Regression / Multinomial Naive Bayes / LinearSVC)

**Classifiers compared:**

| Classifier | Spam Precision | Spam Recall | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.99 | 0.92 | 98% |
| Multinomial Naive Bayes | 0.99 | 0.88 | 98% |
| **LinearSVC** | **0.96** | **0.97** | **99%** |

### Train
```bash
uv run python -m src.tasks.chapter_03.task_4.spam_classifier
```
Downloads the SpamAssassin public corpus (~2.5 MB) on first run to `data/spam_assassin/`.
Trains all three classifiers, saves the best to: `models/model_task4_<acc>pct.joblib`

### Test
```bash
uv run python -m src.tasks.chapter_03.task_4.test_model
```
Loads the saved pipeline, evaluates on the 20% held-out split. Reports accuracy, precision, recall, classification report, and confusion matrix.

> **Note:** The test script must be run after training so the corpus is already downloaded.

---

## Project Structure

```
.
├── data/
│   └── spam_assassin/          # Downloaded SpamAssassin corpus (task 4)
├── models/                     # Saved models (created on first training run)
│   ├── model_task1_<acc>pct.joblib
│   ├── model_task2_<acc>pct.joblib
│   ├── model_task3_<acc>pct.joblib
│   └── model_task4_<acc>pct.joblib
├── src/tasks/chapter_03/
│   ├── task_1/
│   │   ├── mnist_classifier.py   # Train
│   │   └── test_model.py         # Test
│   ├── task_2/
│   │   ├── mnist_augmented.py    # Train
│   │   ├── test_model.py         # Test
│   │   ├── original_images/      # Sample MNIST images
│   │   └── transformed_images/   # Shifted versions
│   ├── task_3/
│   │   ├── titanic_classifier.py # Train
│   │   └── test_model.py         # Test
│   └── task_4/
│       ├── spam_classifier.py    # Train
│       └── test_model.py         # Test
└── pyproject.toml
```
