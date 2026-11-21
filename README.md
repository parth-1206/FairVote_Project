# FairVote – An ML-Based Fair and Transparent Voting System with Election Anomaly Detection

FairVote is a Machine Learning-based voting system designed to provide a fair, transparent, and reliable election process. The project integrates a Flask-based web application, MySQL database, and Machine Learning algorithms to detect potentially anomalous election patterns and analyze fairness across different geographical regions.

The system provides separate interfaces for voters and administrators, supports secure voter authentication, prevents duplicate voting, and presents election statistics and anomaly detection results through an administrator dashboard.

## 🚀 Key Features

### 1. Voter Management
- Voter registration and login.
- Secure password hashing.
- Role-based authentication.
- Voter dashboard.
- View active elections and candidates.
- Cast votes.
- Duplicate-vote prevention.
- Voting confirmation.

### 2. Admin Dashboard
- Secure administrator login.
- View registered voters.
- Monitor total votes.
- Calculate voter turnout.
- View candidate-wise voting results.
- Monitor election status.
- View ML-detected anomalies.

### 3. Machine Learning – Anomaly Detection
- Isolation Forest algorithm.
- Election booth anomaly detection.
- Turnout percentage analysis.
- Candidate vote-share analysis.
- Rejected vote-rate analysis.
- Anomaly score generation.
- Confusion matrix and classification report.

### 4. Fairness Analysis
Uses Fairlearn to evaluate differences in model outcomes across geographical regions.

Metrics:
- Selection Rate
- False Positive Rate
- True Positive Rate
- Selection Rate Difference
- Selection Rate Ratio
- Group-wise fairness comparison

### 5. Data Visualization
- Anomaly rate by region.
- Anomaly score distribution.
- Normal vs. anomalous records.
- Fairlearn selection-rate comparison.
- True Positive Rate and False Positive Rate graphs.
- Confusion matrix visualization.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Database | MySQL |
| Machine Learning | Scikit-learn |
| Anomaly Detection | Isolation Forest |
| Fairness Analysis | Fairlearn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Authentication | Werkzeug |
| Development | VS Code |

---

## 📁 Project Structure

```text
FairVote/
│
├── ML/
│   │
│   ├── .venv/
│   │
│   ├── data/
│   │   ├── graphs/
│   │   ├── anomaly_rate_by_region.png
│   │   ├── anomaly_results.csv
│   │   ├── election_data.csv
│   │   ├── fairlearn_results.csv
│   │   └── fairness_results.csv
│   │
│   ├── models/
│   │
│   ├── notebooks/
│   │
│   ├── src/
│   │   ├── batch_prediction.py
│   │   ├── confusion_matrix.py
│   │   ├── eda.py
│   │   ├── fairlearn_analysis.py
│   │   ├── fairness_analysis.py
│   │   ├── generate_dataset.py
│   │   ├── generate_graph.py
│   │   ├── predict_anomaly.py
│   │   ├── trai_model.py
│   │   ├── validate_dataset.py
│   │   └── visualize_result.py
│   │
│   └── anomaly_detection.py
│
├── webApp/
│   │
│   ├── Database/
│   │   └── db.py
│   │
│   ├── routes/
│   │   ├── admin.py
│   │   ├── auth.py
│   │   └── voting.py
│   │
│   ├── static/
│   │
│   ├── templates/
│   │   ├── admin_dashboard.html
│   │   ├── dashboard.html
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── vote_success.html
│   │   ├── vote.html
│   │   └── voter_dashboard.html
│   │
│   └── app.py
│
└── README.md
```

---

## 🧠 Machine Learning Implementation

### Isolation Forest

FairVote uses Isolation Forest, an unsupervised Machine Learning algorithm, to identify unusual patterns in election booth data.

The model analyzes the following features:

- `turnout_percentage`
- `candidate_A_share`
- `candidate_B_share`
- `candidate_C_share`
- `candidate_D_share`
- `rejected_vote_rate`

### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Isolation Forest |
| Number of estimators | 200 |
| Contamination | 0.05 |
| Random state | 42 |

### Prediction Interpretation

| Prediction | Meaning |
|---|---|
| 1 | Normal |
| -1 | Anomaly |

The predictions are converted into binary labels:

- 0 – Normal
- 1 – Anomalous

The model generates anomaly scores and saves its results in `data/anomaly_results.csv`.

### Model Evaluation

The model is evaluated using:
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report

**Note:** The current anomaly detection experiments use synthetic election booth data. A detected anomaly indicates an unusual statistical pattern, not confirmed election fraud.

---

## ⚖️ Fairness Analysis Using Fairlearn

FairVote evaluates whether anomaly detection outcomes vary across different geographical regions.

The current analysis considers:

- North
- South
- East
- West

The system calculates:

1. Selection Rate
2. False Positive Rate
3. True Positive Rate
4. Difference between groups
5. Ratio between groups

The results are saved in CSV files and visualized through graphs.

---

## 🗄️ Database Design

FairVote uses MySQL for storing and managing voting-related information.

**Database Name:** `fairvote`

### Database Tables

| Table | Description |
|---|---|
| users | Stores voter and admin information |
| elections | Stores election details |
| candidates | Stores candidate information |
| votes | Stores submitted votes |

### Database Relationships

- One election can contain multiple candidates.
- One election can contain multiple votes.
- Each vote belongs to one voter.
- Each vote is associated with one candidate and one election.
- A voter can cast only one vote per election.

---

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/parth-1206/FairVote_Project.git
```

```bash
cd FairVote
```

### Step 2: Create a Virtual Environment

Create a virtual environment inside the ML folder:

```bash
cd ML
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install flask
pip install mysql-connector-python
pip install pandas numpy
pip install scikit-learn
pip install fairlearn
pip install matplotlib seaborn
pip install werkzeug
```

### Step 4: Configure MySQL

Open MySQL Workbench and execute:

CREATE DATABASE fairvote;
```

Configure your MySQL credentials in:

```text
webApp/Database/db.py
```

Make sure the required tables are created before running the application.

Use environment variables for database credentials. Do not upload passwords or secret keys to GitHub.

### Step 5: Run Machine Learning Module

Navigate to the ML folder:

```powershell
cd ML
```

Run the dataset generation script:

```powershell
python src/generate_dataset.py
```

Run model training:

```powershell
python src/trai_model.py
```

Run anomaly prediction:

```powershell
python src/predict_anomaly.py
```

Run fairness analysis:

```powershell
python src/fairness_analysis.py
```

Run Fairlearn analysis:

```powershell
python src/fairlearn_analysis.py
```

Generate visualizations:

```powershell
python src/generate_graph.py
```

The generated results are stored in the `data/` directory.

*Run scripts according to their data dependencies and use the actual script responsible for producing the CSV consumed by the next step.*

### Step 6: Run the Flask Web Application

Open a new terminal.

Navigate to the web application directory:

```powershell
cd webApp
```

Activate the ML virtual environment:

```powershell
..\ML\.venv\Scripts\Activate.ps1
```

Run the Flask application:

```powershell
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000/
```

---

## 🔄 Project Workflow

```text
             Voter Registration
                     |
                     v
                User Login
                     |
                     v
               Active Election
                     |
                     v
              Candidate Selection
                     |
                     v
                  Cast Vote
                     |
                     v
                MySQL Database
                     |
                     v
              Election Statistics
                     |
                     v
             Admin Dashboard


              ML DATASET
                  |
                  v
             Data Processing
                  |
                  v
             Isolation Forest
                  |
                  v
            Anomaly Detection
                  |
                  v
             Fairness Analysis
                  |
                  v
             Result Generation
                  |
                  v
             Admin Dashboard
```

---

## 🎯 Project Objectives

1. Develop a secure web-based voting application.
2. Implement voter authentication and role-based access.
3. Prevent duplicate voting.
4. Apply Isolation Forest for election anomaly detection.
5. Evaluate fairness across geographical regions.
6. Provide an administrator dashboard for election monitoring.

---


## 🔮 Future Scope

- Integration of ML with real-time voting activity.
- Real-time anomaly notifications.
- Advanced election activity monitoring.
- Comparison of multiple anomaly detection algorithms.
- Improved fairness evaluation.
- Interactive election analytics.
- Cloud deployment.
- Enhanced security and audit mechanisms.

---

## ⚠️ Disclaimer

FairVote is an academic project developed for educational, research, and demonstration purposes. Its current ML experiments use synthetic election data. Anomaly detection results should not be treated as proof of election fraud.
---

⭐ If you find this project interesting, consider giving the repository a star!
