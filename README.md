\# FraudIntel — End-to-End Fraud Detection \& Monitoring Platform



FraudIntel is an end-to-end fraud detection and analytics platform designed to demonstrate practical machine learning and data engineering skills.



The project combines transaction data processing, machine learning, API-based predictions, simulated transaction streaming, model monitoring and an interactive dashboard.



It explores how transaction data can be transformed into actionable fraud intelligence through an integrated data and machine learning workflow.



\## Project Overview



FraudIntel brings together several components of a modern fraud detection system:



1\. \*\*Data engineering:\*\* Transaction data ingestion, validation and processing.

2\. \*\*Data processing:\*\* PySpark, Spark SQL, DuckDB and Parquet-based workflows.

3\. \*\*Machine learning:\*\* LightGBM-based fraud prediction and threshold-based classification.

4\. \*\*Experiment tracking:\*\* MLflow for tracking model development and experiments, including earlier CatBoost experiments.

5\. \*\*Backend services:\*\* FastAPI prediction endpoints and supporting application workflows.

6\. \*\*Streaming simulation:\*\* Simulated incoming transactions and prediction workflows.

7\. \*\*Monitoring:\*\* Data quality, model performance and drift monitoring using Evidently.

8\. \*\*Dashboard:\*\* A React-based interface for interacting with the platform and viewing operational insights.



\## Technology Stack



| Area | Technologies |

|---|---|

| Programming | Python, JavaScript |

| Data engineering | PySpark, Spark SQL, DuckDB |

| Data processing | Pandas, PyArrow, Parquet |

| Machine learning | LightGBM, CatBoost, Scikit-learn |

| Experiment tracking | MLflow |

| Model explainability | SHAP |

| Model monitoring | Evidently |

| Backend | FastAPI, Uvicorn |

| Frontend | React, Vite, Recharts |

| Testing | Pytest, pytest-cov |

| Containerisation | Docker, Docker Compose |



\## Key Features



\- Transaction data ingestion and validation.

\- Data exploration and profiling.

\- Data integrity and Parquet validation.

\- Distributed data processing workflows.

\- DuckDB and SQL-based data exploration.

\- LightGBM fraud prediction.

\- Threshold-based classification.

\- API-based prediction workflows.

\- Batch transaction assessment.

\- Simulated transaction streaming.

\- Model and data monitoring.

\- Drift and performance monitoring.

\- Prediction explainability.

\- Interactive fraud monitoring dashboard.

\- Automated tests covering core application components.



\## Machine Learning and Experiment Tracking



LightGBM is the current model used in the fraud prediction and inference workflow.



The project also uses MLflow to track machine learning development and experiment history, including earlier CatBoost experiments.



Historical experimental artifacts are distinguished from the model currently used for inference.



Model retraining and further optimisation are part of the ongoing development roadmap.



\## Project Architecture



```text

Transaction Data

&#x20;     |

&#x20;     v

Data Ingestion and Validation

&#x20;     |

&#x20;     v

Data Processing

(PySpark / Spark SQL / DuckDB)

&#x20;     |

&#x20;     v

Parquet Data Storage

&#x20;     |

&#x20;     v

Machine Learning

(LightGBM)

&#x20;     |

&#x20;     v

Prediction and Inference API

(FastAPI)

&#x20;     |

&#x20;     v

Monitoring and Streaming Simulation

&#x20;     |

&#x20;     v

React Dashboard

```



MLflow supports experiment tracking, while Evidently supports data and model monitoring workflows.



\## Project Structure



```text

FraudIntel/

│

├── app/                  # Backend application and API modules

├── src/                  # Core source code

├── config/               # Application configuration

├── configs/              # Additional configuration files

├── data/                 # Local datasets and processed data

├── frontend/             # React and Vite dashboard

├── models/               # Active model files

├── artifacts/            # Model and inference artifacts

├── monitoring/           # Monitoring components

├── notebooks/            # Data exploration and experiments

├── scripts/              # Ingestion and validation scripts

├── reports/              # Generated reports

├── tests/                # Automated tests

│

├── docker-compose.yml

├── requirements.txt

├── pytest.ini

└── README.md

```



\## Getting Started



\### Prerequisites



\- Python environment compatible with the project dependencies.

\- Node.js and npm.

\- Java for local PySpark execution.

\- Git.

\- Docker Desktop (optional).



\### 1. Clone the repository



```bash

git clone https://github.com/Denford7/FraudIntel.git

cd FraudIntel

```



\### 2. Set up the Python environment



```bash

python -m venv .venv

```



Activate the environment.



\*\*Windows PowerShell:\*\*



```powershell

.venv\\Scripts\\Activate.ps1

```



Install the dependencies:



```bash

pip install -r requirements.txt

```



\### 3. Start the backend



From the project root, run:



```bash

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

```



The backend will be available at:



```text

http://127.0.0.1:8000

```



FastAPI interactive documentation:



```text

http://127.0.0.1:8000/docs

```



\### 4. Start the frontend



Open a second terminal:



```powershell

cd frontend

npm install

npm run dev

```



Vite will display the local dashboard URL, normally:



```text

http://localhost:5173

```



Keep both the backend and frontend terminals running during local development.



\### 5. Run the tests



From the project root:



```bash

pytest -v

```



For test coverage:



```bash

pytest --cov=app --cov-report=term-missing

```



\## Data Availability



Large datasets and generated data files are maintained locally rather than committed to the Git repository.



To reproduce the data workflows, obtain the required source dataset separately and follow the ingestion, profiling and validation scripts in the `scripts/` directory.



\## Testing and Validation



The project includes automated tests covering:



\- API functionality.

\- Data integrity.

\- Data pipeline execution.

\- Data quality reporting.

\- Logging configuration.

\- Monitoring persistence.

\- Parquet integrity and values.

\- Raw data validation.

\- Transaction stream simulation.



\## Development Roadmap



FraudIntel is being developed incrementally.



\### Implemented



\- Data ingestion and validation workflows.

\- PySpark, Spark SQL and DuckDB data processing.

\- Parquet-based data workflows.

\- LightGBM inference.

\- FastAPI backend.

\- Simulated transaction streaming.

\- Evidently monitoring workflows.

\- React dashboard.

\- Automated tests.

\- MLflow experiment tracking.



\### Upcoming



\- Model retraining and optimisation.

\- Further model evaluation and champion model selection.

\- Retrieval-Augmented Generation (RAG) integration.

\- Cloud deployment using Render.

\- Production deployment documentation.



\## Project Goals



FraudIntel is designed to demonstrate practical capabilities across the data engineering and machine learning lifecycle:



\- Building reproducible data pipelines.

\- Working with distributed data processing.

\- Applying SQL to analytical workflows.

\- Developing fraud detection models.

\- Deploying machine learning through APIs.

\- Simulating transaction processing workflows.

\- Monitoring model and data quality.

\- Building interactive applications around machine learning systems.



\## Author



\*\*Denford Mugadza\*\*



GitHub: \[Denford7](https://github.com/Denford7)



