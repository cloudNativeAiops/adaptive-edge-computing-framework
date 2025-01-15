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

Run different evaluation scenarios:

#### Standard evaluation (3 nodes)
make evaluate
#### Scale-up evaluation (4 nodes)
make evaluate-scale-up
#### Scale-down evaluation (2 nodes)
make evaluate-scale-dow

Each evaluation will:
1. Create Docker containers simulating edge devices
2. Install necessary dependencies
3. Run distributed inference tests
4. Collect comprehensive performance metrics

View the results:  

```bash
make show-results
```


### Performance Metrics

The framework evaluates model performance under different configurations:

#### 1. Standard Configuration (3 nodes)
- Baseline: 2 cores, 2GB memory
- Distributed: 3 nodes
  - Node 1: 1.0 core, 1GB (high)
  - Node 2: 0.6 core, 512MB (medium)
  - Node 3: 0.4 core, 512MB (low)
- Results:
  ```
  Latency reduction: 78.35%
  Throughput improvement: 414.73%
  Scheduling overhead: 10.00ms
  ```

#### 2. Scale-up Configuration (4 nodes)
- Baseline: 3 cores, 3GB memory
- Distributed: 4 nodes
  - Node 1: 1.0 core, 1GB
  - Node 2: 1.0 core, 1GB
  - Node 3: 0.6 core, 512MB
  - Node 4: 0.4 core, 512MB
- Results:
  ```
  Latency reduction: 77.17%
  Throughput improvement: 422.32%
  Scheduling overhead: 10.00ms
  ```README.md

#### 3. Scale-down Configuration (2 nodes)
- Baseline: 1 core, 1GB memory
- Distributed: 2 nodes
  - Node 1: 1.0 core, 1GB
  - Node 2: 0.6 core, 512MB
- Results:
  ```
  Latency reduction: 78.33%
  Throughput improvement: 425.73%
  Scheduling overhead: 10.00ms
  ```

### Key Findings

1. **Performance Improvement**
   - Consistent latency reduction (~78%) across configurations
   - Significant throughput improvement (>400%)
   - Stable scheduling overhead (10ms)

2. **Scalability**
   - Effective resource utilization in all configurations
   - Linear performance scaling with added nodes
   - Minimal overhead increase with node count

3. **Resource Efficiency**
   - Optimal task distribution across nodes
   - Balanced resource utilization
   - Adaptive load balancing

### Configuration

Adjust edge device constraints in `src/docker_manager.py`:

## Project Structure

```
.
├── src/
│ ├── models/ # Model definitions
│ ├── utils/ # Utility functions
│ └── docker_manager.py # Edge device simulation
├── experiments/
│ └── evaluation/ # Performance evaluation
├── data/ # Dataset storage
└── results/ # Evaluation results
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
