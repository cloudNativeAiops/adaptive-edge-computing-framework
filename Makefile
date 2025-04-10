.PHONY: evaluate check-docker baseline-single evaluate-distributed show-results evaluate-scale-up evaluate-scale-down

# infrastructure check
check-docker:
	@echo "Checking Docker environment..."
	@if ! docker info > /dev/null 2>&1; then \
		echo "Error: Docker daemon is not running"; \
		exit 1; \
	fi
	@echo "Docker is ready"

# main evaluation
evaluate: check-docker
	@echo "Running evaluation with 3 nodes..."
	@PYTHONPATH=$(PWD) python experiments/baseline/mobilenet_baseline.py --cpu 2 --memory 2g
	@PYTHONPATH=$(PWD) python experiments/optimized/adaptive_inference.py --nodes 3
	@$(MAKE) show-results

# baseline evaluation
baseline-single:
	@echo "Running single-node baseline evaluation..."
	PYTHONPATH=$(PWD) python experiments/baseline/mobilenet_baseline.py

# standard 3 nodes evaluation
evaluate-distributed:
	@echo "Running distributed system evaluation..."
	PYTHONPATH=$(PWD) python experiments/optimized/adaptive_inference.py

# show comparison results
show-results:
	@echo "\n=== Performance Comparison Report ==="
	@PYTHONPATH=$(PWD) python experiments/generate_report.py

# scale-up evaluation (4 nodes)
evaluate-scale-up: check-docker
	@echo "Running scale-up evaluation with 4 nodes..."
	@PYTHONPATH=$(PWD) python experiments/baseline/mobilenet_baseline.py --cpu 3 --memory 3g
	@PYTHONPATH=$(PWD) python experiments/optimized/adaptive_inference.py --nodes 4
	@$(MAKE) show-results

# scale-down evaluation (2 nodes)
evaluate-scale-down: check-docker
	@echo "Running scale-down evaluation with 2 nodes..."
	@PYTHONPATH=$(PWD) python experiments/baseline/mobilenet_baseline.py --cpu 1 --memory 1g
	@PYTHONPATH=$(PWD) python experiments/optimized/adaptive_inference.py --nodes 2
	@$(MAKE) show-results
