# A3-PS003-TC419
# DRISHTI — Edge AI-Powered Traffic Flow Prediction

### Team ResolveX | AIORI Hackathon 3 | PS-003

**Federated Traffic-Flow Prediction with Communication-Cost Evaluation**

> **DRISHTI:** Predict smarter. Learn locally. Measure communication.

## 1. Overview

DRISHTI is an Edge AI-oriented traffic forecasting project that explores how multiple traffic intersections can collaboratively train a machine learning model without routinely sharing their raw traffic data with a central server.

The project uses **Federated Learning (FL)** and **Federated Averaging (FedAvg)** to study traffic-flow prediction across multiple simulated intersection clients.

Instead of relying exclusively on centralized training, each client trains a local model using its assigned traffic data and shares model updates with a central coordinator. The coordinator aggregates these updates to improve the global model.

DRISHTI evaluates the trade-off between traffic prediction accuracy and communication overhead.

The project follows the requirements of problem statement PS-003 under the Smart Cities category.

## 2. Problem Statement

**PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation**

Traditional centralized traffic forecasting may require traffic observations from roadside sensors and cameras to be transmitted to a central cloud for model training.

This approach can introduce several challenges:

- High network communication overhead.
- Continuous transmission of raw traffic observations.
- Concentration of potentially sensitive travel information.
- Differences in traffic patterns across intersections.
- Difficulty balancing prediction quality with communication costs.

DRISHTI explores whether federated training can provide useful traffic predictions while reducing the need to centralize raw training data.

## 3. Proposed Solution

DRISHTI uses a simulated federated learning environment consisting of three to five virtual intersection clients.

Each client receives a portion of a public traffic dataset and trains a compact forecasting model locally.

The clients send their model updates to a central coordinator, which aggregates them using Federated Averaging. The resulting global model is distributed back to the clients for subsequent training rounds.

The system compares centralized and federated training under IID and non-IID data distributions.

The evaluation focuses on:

- Prediction accuracy using MAE and RMSE.
- Communication bytes transmitted during training.
- Model convergence across communication rounds.
- Performance differences between IID and non-IID client distributions.
- Reproducibility across three random seeds.

## 4. How DRISHTI Works

The system follows this workflow:

```text
        Public Traffic Dataset
                  |
                  v
       Data Preprocessing
                  |
                  v
      Chronological Data Split
                  |
                  v
       Client Data Partitioning
                  |
          +-------+-------+
          |       |       |
          v       v       v
       Client 1 Client 2 Client N
          |       |       |
          v       v       v
       Local Model Training
          |       |       |
          +-------+-------+
                  |
                  v
       Model Updates Transmitted
                  |
                  v
       Central Federated Server
                  |
                  v
          FedAvg Aggregation
                  |
                  v
          Updated Global Model
                  |
                  v
       Redistribute Global Model
                  |
                  v
        Next Training Round
                  |
                  v
       Model Evaluation & Report
```

The training loop repeats for multiple communication rounds until the configured training procedure is complete.

Raw training records remain assigned to their respective clients during federated training; model updates are exchanged instead.

## 5. Core Components

### 5.1 Public Traffic Dataset

The project uses a public traffic dataset containing historical traffic observations.

The dataset is preprocessed and divided chronologically into training and testing portions to avoid using future observations to train a model evaluated on earlier-held-out data.

**Dataset:** Add the actual dataset name and source URL here once finalized.

### 5.2 Simulated Intersection Clients

The dataset is partitioned across three to five virtual clients, with each client representing an independent traffic intersection.

Each client has its own local training data and participates in the federated learning process.

This setup allows the project to simulate distributed traffic environments without requiring physical roadside hardware.

### 5.3 Local Traffic Forecasting Model

Each client trains the same compact forecasting model using its local training data.

The model learns relationships between historical traffic observations and the target traffic-flow value.

The selected model architecture should be documented here after implementation is finalized.

### 5.4 Federated Coordinator

The coordinator manages communication rounds, distributes the current global model, receives client updates, and aggregates them.

It also supports the collection of training and evaluation metrics.

### 5.5 Communication-Cost Evaluator

DRISHTI measures the amount of data exchanged during federated training.

The evaluator can record:

- Client upload bytes per round.
- Client download bytes per round.
- Total upload and download bytes.
- Total communication bytes across training rounds.
- Number of communication rounds.

The measurement convention should remain consistent across experiments.

### 5.6 Evaluation Module

The evaluation module compares the centralized and federated approaches using prediction accuracy, communication overhead, and convergence behavior.

## 6. Federated Averaging (FedAvg)

Federated Averaging is the aggregation algorithm used to combine model updates from participating clients.

Each client trains a local model and returns its updated parameters. The central coordinator computes a weighted average of these parameters to create the next global model.

The aggregation formula is:

\[
w_{\text{global}}=\sum_{k=1}^{K}\frac{n_k}{N}w_k
\]

Where:

- \(K\) is the number of participating clients.
- \(w_k\) represents the updated model parameters from client \(k\).
- \(n_k\) is the number of training examples at client \(k\).
- \(N\) is the total number of training examples across participating clients.

This weighted aggregation allows clients with different numbers of training examples to contribute proportionally to the global model.

### FedAvg Pseudocode

```text
Initialize global model

For each communication round:

    Distribute global model to participating clients

    For each client:
        Train model using local traffic data
        Generate updated model parameters
        Measure communication bytes
        Send model update to coordinator

    Aggregate client models using FedAvg

    Evaluate the updated global model

    Save prediction and communication metrics

Repeat for configured communication rounds
```

## 7. IID vs. Non-IID Experiments

DRISHTI evaluates the impact of different client data distributions.

| Aspect | IID Distribution | Non-IID Distribution |
|---|---|---|
| Data distribution | Clients have broadly similar distributions | Clients have different distributions |
| Traffic patterns | More comparable across clients | Potentially heterogeneous across clients |
| Learning challenge | Generally less heterogeneous | May make global training more challenging |
| Research objective | Establish a baseline | Study the effect of client heterogeneity |

Comparing these settings helps reveal how differences between intersections influence prediction accuracy and federated convergence.

## 8. Centralized vs. Federated Learning

DRISHTI compares two training approaches.

**Centralized learning**

The model is trained using the combined training data under the chosen experimental protocol.

**Federated learning**

Clients train locally and share model updates with the coordinator, which aggregates them using FedAvg.

The comparison examines:

- MAE and RMSE.
- Training and evaluation procedures.
- Communication overhead.
- Convergence across training rounds.
- Sensitivity to IID and non-IID distributions.

The centralized and federated approaches should use comparable preprocessing, forecasting targets, and test evaluation procedures wherever possible.

Federated learning is not assumed to outperform centralized learning automatically. Its performance must be determined experimentally.

## 9. Evaluation Metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted traffic-flow values.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

A lower MAE generally indicates better prediction accuracy.

### Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE penalizes larger prediction errors more heavily than MAE.

### Communication Cost

Total communication cost is calculated by summing the measured transmitted bytes across participating clients and communication rounds.

\[
B_{\text{total}}=B_{\text{upload}}+B_{\text{download}}
\]

Report upload and download bytes separately as well as their total.

### Convergence

Convergence is studied by tracking evaluation performance across communication rounds.

The experiments are repeated with three random seeds to assess variability and improve reproducibility.

## 10. Experimental Methodology

The experimental procedure follows these steps:

1. Select and load one public traffic dataset.
2. Preprocess the traffic observations.
3. Create chronological training and testing splits.
4. Partition the training data among three to five simulated clients.
5. Construct IID and non-IID client distributions.
6. Train the centralized baseline.
7. Train the federated model using FedAvg.
8. Measure communication bytes during each federated round.
9. Evaluate MAE and RMSE.
10. Analyze convergence behavior.
11. Repeat experiments across three random seeds.
12. Compare the results and prepare the final evaluation report.

The same evaluation conventions should be applied consistently across experiments.

## 11. Technology Stack

Update this section to reflect the libraries actually used in the implementation.

Potential tools include:

- **Python** — Main programming language.
- **NumPy** — Numerical computations.
- **Pandas** — Dataset processing.
- **Scikit-learn** — Preprocessing and evaluation utilities.
- **PyTorch or TensorFlow** — Model training, if used.
- **Matplotlib** — Performance and convergence plots.
- **Git and GitHub** — Version control and team collaboration.

Do not list an optional framework as an implemented dependency unless the project actually uses it.

## 12. Project Structure

A suggested repository structure is:

```text
DRISHTI/
│
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   ├── preprocessing.py
│   ├── client.py
│   ├── model.py
│   ├── server.py
│   ├── fedavg.py
│   ├── communication.py
│   └── evaluation.py
├── experiments/
│   ├── centralized.py
│   ├── federated_iid.py
│   └── federated_non_iid.py
├── results/
│   └── README.md
├── docs/
│   └── architecture.md
└── notebooks/
    └── exploration.ipynb
```

This is a proposed structure. Adapt the paths and filenames to the actual repository rather than creating placeholder files solely to match this diagram.

## 13. Installation and Execution

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd DRISHTI
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project using the actual entry-point script configured in the repository.

For example, if the implementation provides these scripts:

```bash
python experiments/centralized.py
python experiments/federated_iid.py
python experiments/federated_non_iid.py
```

Replace these commands with the verified commands for the final implementation.

**Dataset setup:** Follow the dataset-specific instructions documented in the repository. Do not commit restricted, confidential, or unnecessarily large raw datasets.

## 14. Results and Reporting

The final report should compare centralized and federated training across IID and non-IID distributions.

Include the following table after experiments have been completed:

| Experiment | MAE | RMSE | Communication Bytes |
|---|---:|---:|---:|
| Centralized baseline | To be measured | To be measured | Report applicable transmission scope |
| Federated IID | To be measured | To be measured | To be measured |
| Federated non-IID | To be measured | To be measured | To be measured |

Report actual measurements rather than estimates.

Recommended visualizations include:

- Actual vs. predicted traffic flow.
- MAE and RMSE comparisons.
- Communication bytes per round.
- Cumulative communication cost.
- Convergence curves.
- Results across three random seeds.

Interpret the results in terms of both prediction quality and communication overhead.

## 15. Privacy and Limitations

DRISHTI explores a training approach that avoids routinely sending raw client training records to a central coordinator.

However, sharing model updates does not automatically guarantee formal privacy. Model updates may still reveal information under some circumstances.

The project is a simulation-based prototype and does not claim live camera integration, city-wide deployment, or formal privacy guarantees.

Its findings depend on the selected dataset, forecasting model, client partitioning strategy, training configuration, and communication measurement methodology.

## 16. Expected Outcomes

The project aims to deliver:

- A federated traffic forecasting simulator.
- Centralized and federated model baselines.
- IID and non-IID client experiments.
- A compact forecasting model.
- Communication-cost measurement.
- MAE and RMSE evaluation.
- Convergence analysis.
- Reproducible experiments across three random seeds.
- A final technical evaluation report.

These are project objectives; completed outcomes should be identified separately in the final submission.

## 17. Team ResolveX

**Project:** DRISHTI

**Team Leader:** Devyansh Arya

**Team Member:** Akash Gangwar

**Mentor:** Aman Singh

**Problem Statement:** PS-003 — Federated Traffic-Flow Prediction with Communication-Cost Evaluation

**Event:** AIORI Hackathon 3

GitHub: [Akash Gangwar](https://github.com/akashgangwar68445-alt)

## 18. References

McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. (2017).

*Communication-Efficient Learning of Deep Networks from Decentralized Data.*

Proceedings of Machine Learning Research, Volume 54.

https://proceedings.mlr.press/v54/mcmahan17a.html

---

**DRISHTI — Team ResolveX**

*Building a foundation for communication-aware, federated traffic intelligence.*
