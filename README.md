\# 🛡️ FraudIntel

\### End-to-End Fraud Detection \& Intelligent Monitoring Platform



<p align="center">

&#x20; <strong>Turning transaction data into actionable fraud intelligence.</strong>

</p>



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />

&#x20; <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" />

&#x20; <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react" />

&#x20; <img src="https://img.shields.io/badge/LightGBM-ML-9ACD32" />

&#x20; <img src="https://img.shields.io/badge/PySpark-Data%20Engineering-E25A1C?logo=apachespark" />

&#x20; <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2" />

</p>



\---



\## 🚀 Project Overview



\*\*FraudIntel\*\* is an end-to-end fraud detection and analytics platform built to demonstrate practical Machine Learning Engineering and Data Engineering capabilities.



The platform integrates transaction data processing, machine learning, API-based predictions, simulated transaction streaming, model monitoring and an interactive dashboard.



Rather than treating fraud detection as an isolated classification problem, FraudIntel explores the wider engineering lifecycle required to build and operate a fraud detection system.



\## 🎯 What FraudIntel Does



| Capability | Description |

|---|---|

| 📥 Data ingestion | Transaction data acquisition and ingestion workflows |

| 🔍 Data validation | Raw data validation, integrity checks and profiling |

| ⚙️ Data engineering | PySpark, Spark SQL, DuckDB and Parquet processing |

| 🤖 Fraud prediction | LightGBM-based fraud prediction and threshold classification |

| 🔌 Prediction API | FastAPI backend for model inference |

| 🌊 Streaming simulation | Simulated incoming transaction processing |

| 📊 Model monitoring | Performance, data quality and drift monitoring |

| 🧠 Explainability | SHAP-based model interpretation capabilities |

| 🖥️ Dashboard | React-based fraud intelligence interface |



\## 🏗️ Architecture



```mermaid

flowchart TD

&#x20;   A\["Transaction Data"] --> B\["Data Ingestion"]

&#x20;   B --> C\["Data Validation"]

&#x20;   C --> D\["PySpark / Spark SQL / DuckDB"]

&#x20;   D --> E\["Parquet Data Storage"]

&#x20;   E --> F\["LightGBM Model"]

&#x20;   F --> G\["FastAPI Prediction Service"]

&#x20;   G --> H\["Streaming Simulation"]

&#x20;   G --> I\["Monitoring"]

&#x20;   H --> I

&#x20;   I --> J\["React Dashboard"]

&#x20;   K\["MLflow"] -.-> F

&#x20;   L\["Evidently"] -.-> I

```



\## 🧰 Technology Stack



| Category | Technologies |

|---|---|

| Programming | Python, JavaScript |

| Data Engineering | PySpark, Spark SQL, DuckDB |

| Data Processing | Pandas, PyArrow, Parquet |

| Machine Learning | LightGBM, CatBoost, Scikit-learn |

| Experiment Tracking | MLflow |

| Explainability | SHAP |

| Monitoring | Evidently |

| Backend | FastAPI, Uvicorn |

| Frontend | React, Vite, Recharts |

| Testing | Pytest, pytest-cov |

| Containerisation | Docker, Docker Compose |



\## ✨ Engineering Highlights



\### 1. Data Engineering

\- Structured transaction data ingestion workflows.

\- Raw data validation and integrity verification.

\- Distributed processing using PySpark and Spark SQL.

\- Analytical querying using DuckDB.

\- Parquet-based data storage and validation.



\### 2. Machine Learning

\- LightGBM-based fraud prediction.

\- Threshold-based classification.

\- MLflow experiment tracking.

\- Earlier CatBoost experiments retained in experiment history.

\- SHAP-based explainability capabilities.



\*LightGBM is the current inference model. Historical experimental artifacts are not treated as active production models.\*



\### 3. Backend and Streaming

\- FastAPI prediction services.

\- Transaction batch assessment workflows.

\- Simulated incoming transaction streams.

\- Persistent monitoring and application workflows.



\### 4. Monitoring and Visualisation

\- Model performance monitoring.

\- Data quality and drift monitoring with Evidently.

\- Interactive React dashboard.

\- Recharts-based visualisation.



\## 📁 Project Structure



```text

FraudIntel/

│

├── app/                  # FastAPI application and backend modules

├── src/                  # Core source code

├── config/               # Application configuration

├── configs/              # Additional configurations

├── data/                 # Local transaction datasets

├── frontend/             # React + Vite dashboard

├── models/               # Active model files

├── artifacts/            # Model and inference artifacts

├── monitoring/           # Monitoring components

├── notebooks/            # Data exploration and experiments

├── scripts/              # Ingestion and validation workflows

├── reports/              # Generated reports

├── tests/                # Automated tests

│

├── docker-compose.yml

├── requirements.txt

├── pytest.ini

└── README.md

```



\## ⚡ Getting Started



\### Prerequisites



\- Python environment compatible with project dependencies.

\- Node.js and npm.

\- Java for PySpark.

\- Git.

\- Docker Desktop (optional).



\### 1. Clone the repository



```bash

git clone https://github.com/Denford7/FraudIntel.git

cd FraudIntel

```



\### 2. Install Python dependencies



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

```



\### 3. Start the backend



From the project root:



```bash

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

```



Backend: `http://127.0.0.1:8000`



API documentation: `http://127.0.0.1:8000/docs`



\### 4. Start the frontend



Open another terminal:



```powershell

cd frontend

npm install

npm run dev

```



Dashboard: `http://localhost:5173`



\### 5. Run automated tests



From the project root:



```bash

pytest -v

```



Coverage report:



```bash

pytest --cov=app --cov-report=term-missing

```



\## 📊 Experiment Tracking \& Monitoring



\*\*MLflow\*\* supports experiment tracking and model development history, including earlier CatBoost experiments.



\*\*Evidently\*\* supports data quality and model monitoring workflows, helping assess changes in transaction data and model behaviour.



\## 🗺️ Development Roadmap



| Component | Status |

|---|---|

| Data ingestion and validation | ✅ Implemented |

| PySpark and SQL data processing | ✅ Implemented |

| Parquet workflows | ✅ Implemented |

| LightGBM inference | ✅ Implemented |

| FastAPI backend | ✅ Implemented |

| Streaming simulation | ✅ Implemented |

| Evidently monitoring | ✅ Implemented |

| React dashboard | ✅ Implemented |

| MLflow experiment tracking | ✅ Implemented |

| Model retraining and optimisation | 🔄 Upcoming |

| RAG integration | 📌 Planned |

| Render cloud deployment | 📌 Planned |



\## 🧪 Testing



The project includes automated tests covering:



\- API functionality

\- Data integrity

\- Data pipeline execution

\- Data quality reporting

\- Logging configuration

\- Monitoring persistence

\- Parquet integrity and values

\- Raw data validation

\- Transaction stream simulation



\## 📦 Data Availability



Large transaction datasets and generated data files are maintained locally and are not committed to this repository.



Obtain the required source dataset separately and follow the ingestion and validation workflows in `scripts/`.



\## 👨‍💻 Author



\*\*Denford Mugadza\*\*



\[!\[GitHub](https://img.shields.io/badge/GitHub-Denford7-181717?logo=github)](https://github.com/Denford7)



\---



<p align="center">

&#x20; <strong>FraudIntel — Engineering intelligent fraud detection workflows.</strong>

</p>



