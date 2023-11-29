# QMLvsML

**Objective**: This repository aims to compare the performance of parametrized quantum circuits against classical machine learning models.

## How to Reproduce the Results

- The repository contains 6 notebooks dedicated to training Neural Networks (NN) for regression tasks on the California Housing dataset.
- Frequency generator and data scaler functions are imported from the `utils` file for use in the Fourier_model.
- Exact inversion is implemented using two custom functions, alternatively importable from `utils`.
- Neural Networks are trained with the Fourier Model, exploring various configurations.
- A Neural Network using the ReLU model is defined, varying epoch sizes and optimizers, without specific optimization efforts.

## QML - Quantum Machine Learning

**Purpose**: To explore quantum machine learning as an alternative to classical ML models, with a focus on benchmarking quantum learning models for effective predictions on diverse datasets.

### Datasets Explored

- One-dimensional input.
- A curated clean dataset.
- The California Housing dataset.
- Dataset of compound material properties.

### Implemented Models

- **Neural Network**: Basic and tuned versions.
- **Classical Surrogate Model**: Mimicking quantum learning models based on [this paper](https://arxiv.org/pdf/2206.11740.pdf).
- **Quantum Model**: Strongly entangled template from PennyLane.
- **Classical Surrogate**: With three different encoding strategies, detailed in [this research](https://arxiv.org/pdf/2206.12105.pdf).

### Additional Context

- **Quantum Computing in PDE Simulations**: Recent advances have spurred interest in using quantum computing for solving PDEs in simulations.
- **Insights from Key Research**: Research by M. Ali and M. Kabel [1] raises questions about the near-term applicability of quantum computing for PDEs, considering NISQ limitations.
- **Internship Focus**: Investigating Quantum Machine Learning for material property prediction regression tasks, an alternative approach to solving PDEs with quantum computing.

### Model Analysis

- **Model Selection**: Identifying effective quantum learning models showcasing quantum advantage.
- **Dataset Generation**: Utilizing classical PDE solvers for comprehensive dataset generation.
- **Benchmarking**: Comparing the quantum model's performance to classical ML and PDE solutions.

### My Contribution

- **Reproduce**: Mimic the input-output relationship of re-uploading quantum learning models using Neural Networks and Fourier Series models.
- **Evaluate**: Assess the performance of Neural Networks and compare with various benchmarks.
- **Assess**: Explore the potential quantum advantage in using quantum learning models for regression tasks.

### Resources

- A curated playlist of talks and conferences on QML.
- Adaptations and feedback on Pennylane Tutorials: [Pennylane QML Demonstrations](https://pennylane.ai/qml/demonstrations/quantum-machine-learning).
- A list of relevant courses in quantum computing.
