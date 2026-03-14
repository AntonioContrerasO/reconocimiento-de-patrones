.PHONY: help train-all test-all plots \
        train1 train2 train3 train4 \
        test1 test2 test3 test4

# ── default ──────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Chapter 03 — ML Exercises"
	@echo ""
	@echo "  Training"
	@echo "    make train1   MNIST KNN classifier (>97% accuracy)"
	@echo "    make train2   MNIST KNN + data augmentation"
	@echo "    make train3   Titanic logistic regression"
	@echo "    make train4   Spam classifier (SpamAssassin corpus)"
	@echo "    make train-all  Run all training scripts in order"
	@echo ""
	@echo "  Testing (requires trained model)"
	@echo "    make test1    Evaluate task-1 model — metrics + confusion matrix"
	@echo "    make test2    Evaluate task-2 model — metrics + confusion matrix"
	@echo "    make test3    Evaluate task-3 model — metrics + confusion matrix"
	@echo "    make test4    Evaluate task-4 model — metrics + confusion matrix"
	@echo "    make test-all   Run all test scripts in order"
	@echo ""
	@echo "  Plots"
	@echo "    make plots      Generate metric plots for all trained models"
	@echo ""

# ── training ─────────────────────────────────────────────────────────────────
train1:
	uv run python -m src.tasks.chapter_03.task_1.mnist_classifier

train2:
	uv run python -m src.tasks.chapter_03.task_2.mnist_augmented

train3:
	uv run python -m src.tasks.chapter_03.task_3.titanic_classifier

train4:
	uv run python -m src.tasks.chapter_03.task_4.spam_classifier

train-all: train1 train2 train3 train4

# ── testing ──────────────────────────────────────────────────────────────────
test1:
	uv run python -m src.tasks.chapter_03.task_1.test_model

test2:
	uv run python -m src.tasks.chapter_03.task_2.test_model

test3:
	uv run python -m src.tasks.chapter_03.task_3.test_model

test4:
	uv run python -m src.tasks.chapter_03.task_4.test_model

test-all: test1 test2 test3 test4

# ── plots ─────────────────────────────────────────────────────────────────────
plots:
	uv run python -m src.tasks.chapter_03.plot_metrics
