.PHONY: evaluate check-docker baseline-single evaluate-distributed show-results

# infrastructure check
check-docker:
	@echo "Checking Docker environment..."
	@if ! docker info > /dev/null 2>&1; then \
		echo "Error: Docker daemon is not running"; \
		exit 1; \
	fi
	@echo "Docker is ready"

# main evaluation
evaluate: check-docker baseline-single evaluate-distributed show-results

# baseline evaluation
baseline-single:
	@echo "Running single-node baseline evaluation..."
	PYTHONPATH=$(PWD) python experiments/evaluation/baseline_single.py

# distributed system evaluation
evaluate-distributed:
	@echo "Running distributed system evaluation..."
	PYTHONPATH=$(PWD) python experiments/evaluation/performance_evaluation.py

# show comparison results
show-results:
	@echo "\n=== Baseline Single Node Results ==="
	@PYTHONPATH=$(PWD) python src/utils/show_metrics.py --metrics-file results/evaluation/baseline_metrics.json
	@echo "\n=== Distributed System Results ==="
	@PYTHONPATH=$(PWD) python src/utils/show_metrics.py --metrics-file results/evaluation/performance_metrics.json
	@echo "\n=== Performance Comparison Report ==="
	@PYTHONPATH=$(PWD) python experiments/evaluation/generate_report.py
