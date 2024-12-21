.PHONY: setup test clean docker-build docker-up docker-down experiment-baseline experiment-adaptive init

# Initial setup
init:
	@echo "Creating necessary directories..."
	mkdir -p results/baseline results/optimized results/scenarios
	mkdir -p src/config
	@echo "Checking Python environment..."
	which python3 || (echo "Please install Python 3" && exit 1)
	python3 -m pip install --upgrade pip

# Environment setup
setup: init
	@echo "Installing Python dependencies..."
	python3 -m pip install -r requirements.txt
	@echo "Setup complete! 🚀"

# Virtual environment setup (for macOS)
venv:
	python3 -m venv venv
	@echo "Virtual environment created. Run 'source venv/bin/activate' to activate it"

# Development setup with virtual environment
dev-setup: venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Development environment setup complete! 🚀"

# Export Python path
export PYTHONPATH := $(PYTHONPATH):$(shell pwd)

# Docker commands
docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

# Test commands
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html

# Experiment commands
DOCKER_SOCKET := /var/run/docker.sock
DOCKER_CHECK := $(shell docker info >/dev/null 2>&1 && echo 1 || echo 0)

check-docker:
	@echo "Checking Docker connection..."
	@if ! docker info > /dev/null 2>&1; then \
		echo "Error: Docker is not running or not accessible."; \
		echo "Please:"; \
		echo "1. Ensure Docker Desktop is running"; \
		echo "2. Check Docker Desktop settings"; \
		echo "3. Try restarting Docker Desktop"; \
		exit 1; \
	fi
	@echo "Docker connection successful ✅"

experiment-baseline: check-docker
	DOCKER_HOST=$(DOCKER_HOST) PYTHONPATH=$(shell pwd) python experiments/baseline/mobilenet_baseline.py

experiment-adaptive: check-docker
	DOCKER_HOST=$(DOCKER_HOST) PYTHONPATH=$(shell pwd) python experiments/optimized/adaptive_inference.py

experiment-resource: check-docker experiment-cleanup
	DOCKER_HOST=$(DOCKER_HOST) PYTHONPATH=$(shell pwd) python experiments/scenarios/resource_constraint_test.py

experiment-network: check-docker
	DOCKER_HOST=$(DOCKER_HOST) PYTHONPATH=$(shell pwd) python experiments/scenarios/network_latency_test.py

experiment-cleanup:
	@echo "Cleaning up Docker resources..."
	DOCKER_HOST=$(DOCKER_HOST) PYTHONPATH=$(shell pwd) python -c "from src.utils.docker_utils import DockerManager; DockerManager().cleanup()"
	@echo "Cleanup complete ✅"

# Test model partitioning
experiment-partition:
	DOCKER_HOST=$(DOCKER_HOST) PYTHONPATH=$(shell pwd) python experiments/scenarios/partition_test.py

# Test adaptive scheduling
experiment-scheduling:
	DOCKER_HOST=$(DOCKER_HOST) PYTHONPATH=$(shell pwd) python experiments/scenarios/scheduler_test.py

# Test full pipeline
experiment-pipeline: experiment-cleanup
	DOCKER_HOST=$(DOCKER_HOST) PYTHONPATH=$(shell pwd) python experiments/scenarios/pipeline_test.py

# Cleanup commands
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.pyc" -delete
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type d -name "results" -exec rm -r {} +

# Run all experiments
run-all: experiment-baseline experiment-adaptive experiment-resource experiment-network

# Add this at the top of the Makefile
ifeq ($(shell uname),Darwin)
    DOCKER_HOST ?= unix:///$(HOME)/.docker/run/docker.sock
endif
