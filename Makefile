.PHONY: help install train test lint format docker-build docker-run docker-compose-up docker-compose-down clean clean-old-files cleanup

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make train            - Train the ML model"
	@echo "  make test             - Run tests"
	@echo "  make lint             - Run linter"
	@echo "  make format           - Format code with black"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-compose-up - Start services with docker-compose"
	@echo "  make docker-compose-down - Stop services with docker-compose"
	@echo "  make clean            - Clean temporary files"
	@echo "  make clean-old-files  - Remove old/unused files (src/, output/, etc.)"
	@echo "  make cleanup          - Full cleanup (clean + clean-old-files)"

install:
	pip install -r requirements.txt

train:
	python train_model.py

test:
	pytest tests/ -v

lint:
	flake8 app/ ml_pipeline/ --max-line-length=127
	black --check app/ ml_pipeline/

format:
	black app/ ml_pipeline/

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

clean-old-files:
	@echo "Removing old source code (src/)..."
	@rm -rf src/ || echo "src/ not found or already removed"
	@echo "Removing old visualization outputs (output/)..."
	@rm -rf output/ || echo "output/ not found or already removed"
	@echo "Checking artifacts directory..."
	@if [ -f "Scripts/data.csv" ] && [ -f "artifacts/data.csv" ]; then \
		echo "artifacts/data.csv is duplicate, removing artifacts/..."; \
		rm -rf artifacts/ || echo "artifacts/ not found"; \
	else \
		echo "Keeping artifacts/ (data may be needed)"; \
	fi
	@echo "Cleanup complete! Review CLEANUP_GUIDE.md for details."

cleanup: clean clean-old-files
	@echo "Full cleanup completed!"

