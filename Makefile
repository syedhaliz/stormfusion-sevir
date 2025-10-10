
.PHONY: setup lint test train-tiny

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pre-commit install

lint:
	ruff check --fix

test:
	pytest -q

train-tiny:
	python scripts/train.py --config configs/base.yaml --data.tiny=true --max_steps=100
