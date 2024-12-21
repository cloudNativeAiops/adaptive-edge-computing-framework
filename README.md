# Adaptive Edge Computing Framework

This framework implements an adaptive edge computing system for efficient deep learning model inference in resource-constrained environments. It features dynamic resource monitoring, model partitioning, and adaptive task scheduling.

## Features

- **Dynamic Resource Monitoring**: Real-time monitoring of CPU, memory, and network usage
- **Model Partitioning**: Intelligent splitting of deep learning models based on computational requirements
- **Adaptive Task Scheduling**: Resource-aware task distribution across edge nodes
- **Docker-based Edge Simulation**: Configurable edge node environments with resource constraints
- **Comprehensive Testing**: Unit tests and scenario-based experiments

## Project Structure

```
adaptive-edge-computing-framework/
├── src/
│   ├── core/
│   │   ├── resource_monitor.py      # Resource monitoring
│   │   ├── model_partitioner.py     # Model partitioning
│   │   ├── task_scheduler.py        # Adaptive scheduling
│   │   └── model_deployer.py        # Model deployment
│   ├── models/
│   │   └── mobilenet.py            # MobileNet implementation
│   └── utils/
│       ├── docker_utils.py         # Docker management
│       └── metrics.py              # Performance metrics
├── experiments/
│   ├── baseline/                   # Baseline performance
│   ├── optimized/                  # Optimized inference
│   └── scenarios/                  # Test scenarios
└── tests/                          # Unit tests
```

## Getting Started

### Prerequisites

- Python 3.10 - 3.11
- Docker Desktop
- PyTorch 2.1+

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd adaptive-edge-computing-framework
```

2. Create and activate virtual environment:
```bash
make venv
source venv/bin/activate
```

3. Install dependencies:
```bash
make setup
```

### Running Tests

Run all unit tests:
```bash
make test
```

Generate coverage report:
```bash
make test-coverage
```

### Running Experiments

1. Resource Constraint Tests:
```bash
make experiment-resource
```

2. Model Partitioning Tests:
```bash
make experiment-partition
```

3. Adaptive Scheduling Tests:
```bash
make experiment-scheduling
```

4. Full Pipeline Test:
```bash
make experiment-pipeline
```

## Experiment Results

### Resource Constraints
- High Resource (CPU: 1.0, Memory: 1G): ~22ms inference time
- Medium Resource (CPU: 0.5, Memory: 512M): ~23ms inference time
- Low Resource (CPU: 0.2, Memory: 256M): ~40ms inference time

### Model Partitioning
Successfully partitions models into:
- 2-part configuration: [116, 25] layers
- 3-part configuration: [108, 16, 17] layers
- 4-part configuration: [100, 16, 16, 9] layers

### Adaptive Scheduling
Demonstrates effective load balancing and resource-aware task distribution across nodes.

## Development

### Adding New Tests

1. Create test file in `tests/` directory
2. Run tests with coverage:
```bash
make test-coverage
```

### Adding New Experiments

1. Create experiment script in `experiments/scenarios/`
2. Add make command in Makefile
3. Run experiment:
```bash
make experiment-[name]
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Docker team for containerization support
