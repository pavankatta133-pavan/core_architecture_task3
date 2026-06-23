# PlaceMux — Phase 1 Task 3
## Core Architecture — Modular ML Pipeline

---

## 📌 Overview
This project builds a clean, modular, config-driven ML pipeline
for the PlaceMux student-to-job matching platform.
Any model can plug into the same training harness
without rewriting code.

---

## 🎯 Objective
- Establish core modelling architecture
- Build a config-driven train/eval skeleton
- Wire baseline model through the pipeline
- Log metrics to MLflow experiment tracker

---

## 🗂️ Project Structure
---

## ⚙️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/pavankatta133-pavan/core_architecture_task3.git
cd core_architecture_task3
```

### 2. Create virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install libraries
```bash
pip install -r requirements.txt
```

### 4. Run training pipeline
```bash
python train.py
```

---

## 🔧 How to Add a New Model

### Step 1 — Change config.yaml
```yaml
model:
  name: "LogisticRegression"
```

### Step 2 — Add to ml_model.py
```python
elif model_name == "LogisticRegression":
    model = LogisticRegression(random_state=seed)
```

### Step 3 — Run train.py again
No other file needs to change! ✅

---

## 📊 Results

| Split | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Val   | 0.50     | 0.00     | 0.00      | 0.00   |
| Test  | 0.50     | 0.00     | 0.00      | 0.00   |

> Baseline DummyClassifier — every future model must beat this!

---

## 🛠️ Tools & Stack

| Tool | Purpose |
|------|---------|
| Python 3.14 | Core language |
| Pandas | Data loading |
| NumPy | Numerical computing |
| Scikit-learn | ML pipeline & models |
| MLflow | Experiment tracking |
| PyYAML | Config management |
| Git | Version control |

---

## ✅ Definition of Done

- [x] Modular project structure created
- [x] Config-driven pipeline (config.yaml)
- [x] Single training harness for any model
- [x] Baseline model running through pipeline
- [x] Metrics logged to MLflow
- [x] How to add new model documented
- [x] Git committed and pushed to GitHub

---

## 👤 Author
**Pavan**
AI / ML Developer — PlaceMux Phase 1