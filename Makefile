PYTHON := $(shell command -v python3 || command -v python)
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PO := $(VENV)/bin/po

.PHONY: init test app excel lint format typecheck clean dev

init:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	@echo "✅ Environment ready. Activate with: source $(VENV)/bin/activate"

test:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/black --check src tests
	$(VENV)/bin/mypy src
	$(VENV)/bin/bandit -q -r src || true
	$(VENV)/bin/pytest

app:
	$(VENV)/bin/streamlit run "streamlit_app/0_📈_Portfolio_Optimizer.py"

excel:
	$(VENV)/bin/xlwings addin install || true

format:
	$(VENV)/bin/black src tests
	$(VENV)/bin/ruff check --fix src tests

typecheck:
	$(VENV)/bin/mypy src

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/bandit -q -r src || true

dev: app

clean:
	rm -rf $(VENV) .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} +
