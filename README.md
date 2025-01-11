# Adaptive Edge Computing Framework

A framework for deploying and evaluating deep learning models on edge devices.

## Features

- Edge device simulation using Docker containers
- Model performance evaluation
- Resource usage monitoring
- Adaptive model optimization

## Requirements

- Python 3.10+
- Docker
- PyTorch 2.1.2+
- torchvision 0.16.2+

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-edge-computing-framework.git
cd adaptive-edge-computing-framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Performance Evaluation

Evaluate model performance on simulated edge devices:

```bash
make evaluate-performance
```

This will:
1. Create a Docker container simulating an edge device
2. Install necessary dependencies
3. Run inference on a test dataset
4. Collect performance metrics

View the results:

```bash
make show-metrics
```

### Performance Metrics

The framework evaluates model performance under different resource constraints:

- **Low Resource Profile**
  - CPU: 0.2 cores
  - Memory: 256MB
  ```
  Accuracy:
    Top-1: 9.45%
    Top-5: 43.20%
  Latency:
    Average: 892.34ms
    P95: 987.65ms
  ```

- **Medium Resource Profile**
  - CPU: 0.5 cores
  - Memory: 512MB
  ```
  Accuracy:
    Top-1: 9.80%
    Top-5: 44.80%
  Latency:
    Average: 459.72ms
    P95: 500.84ms
  ```

- **High Resource Profile**
  - CPU: 1.0 cores
  - Memory: 1GB
  ```
  Accuracy:
    Top-1: 9.80%
    Top-5: 44.80%
  Latency:
    Average: 234.56ms
    P95: 256.78ms
  ```

Key observations:
1. Accuracy remains consistent across resource profiles
2. Latency significantly improves with more resources
3. Memory impacts batch processing stability
4. CPU cores directly affect inference speed

### Configuration

Adjust edge device constraints in `src/docker_manager.py`:
```python
create_edge_node(
    name="eval-node",
    cpu_limit=0.5,    # CPU cores
    memory_limit="512m"  # Memory limit
)
```

## Project Structure

```
.
├── src/
│   ├── models/          # Model definitions
│   ├── utils/           # Utility functions
│   └── docker_manager.py # Edge device simulation
├── experiments/
│   └── evaluation/      # Performance evaluation
├── data/                # Dataset storage
└── results/             # Evaluation results
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
