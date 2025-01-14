.PHONY: check-docker-full evaluate-all baseline-single evaluate-performance generate-report

# 完整的 Docker 环境检查
check-docker-full:
	@echo "Performing comprehensive Docker environment check..."
	@echo "\nChecking Docker daemon..."
	@if ! docker info > /dev/null 2>&1; then \
		echo "Error: Docker daemon is not running"; \
		echo "Please start Docker and try again"; \
		exit 1; \
	fi
	@echo "Docker daemon is running"
	
	@echo "\nChecking Docker socket..."
	@if [ -e /var/run/docker.sock ]; then \
		echo "Docker socket exists at default location"; \
	elif [ -e ~/.docker/run/docker.sock ]; then \
		echo "Docker socket exists at Docker Desktop location"; \
	else \
		echo "Warning: Docker socket not found at common locations"; \
	fi
	
	@echo "\nChecking Docker permissions..."
	@if groups | grep -q docker; then \
		echo "User is in docker group"; \
	else \
		echo "Warning: User is not in docker group"; \
	fi
	
	@echo "\nTesting Docker functionality..."
	@if docker run --rm hello-world > /dev/null 2>&1; then \
		echo "Successfully ran test container"; \
	else \
		echo "Error: Failed to run test container"; \
		exit 1; \
	fi

# 运行完整评估
evaluate-all: check-docker-full baseline-single evaluate-performance generate-report

# 单机基线测试
baseline-single:
	@echo "Running single-machine baseline evaluation..."
	PYTHONPATH=$(PWD) python experiments/evaluation/baseline_single.py

# 运行性能评估
evaluate-performance: check-docker-full
	@echo "Running distributed performance evaluation..."
	PYTHONPATH=$(PWD) python experiments/evaluation/performance_evaluation.py

# 生成对比报告
generate-report:
	@echo "Generating comparison report..."
	PYTHONPATH=$(PWD) python experiments/evaluation/generate_report.py

# 显示对比结果
show-comparison:
	@echo "Showing comparison results..."
	cat results/evaluation/comparison_report.json
