.PHONY: help install install-dev train test lint format docker-build docker-run docker-compose-up docker-compose-down clean setup-pre-commit run

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev      - Install dependencies with dev tools"
	@echo "  make setup-pre-commit - Setup pre-commit hooks"
	@echo "  make train            - Train the ML model"
	@echo "  make run              - Run the FastAPI application"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Run linter"
	@echo "  make format           - Format code with black and isort"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-compose-up - Start services with docker-compose"
	@echo "  make docker-compose-down - Stop services with docker-compose"
	@echo "  make clean            - Clean temporary files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

setup-pre-commit:
	pre-commit install

train:
	python train_model.py

run:
	@if [ -f "dev/bin/activate" ]; then \
		source dev/bin/activate && python run.py; \
	else \
		python3 run.py; \
	fi

test:
	pytest tests/ -v

lint:
	flake8 app/ ml_pipeline/ --max-line-length=100 --extend-ignore=E203,W503
	black --check app/ ml_pipeline/ tests/
	@if command -v mypy > /dev/null 2>&1; then \
		mypy app/ ml_pipeline/ --ignore-missing-imports; \
	else \
		echo "mypy not installed, skipping type checking"; \
	fi

format:
	black app/ ml_pipeline/ tests/
	isort app/ ml_pipeline/ tests/

docker-build:
	docker build -t customer-churn-api:latest .

docker-run:
	docker run -p 8000:8000 customer-churn-api:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/


