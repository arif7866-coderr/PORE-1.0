# PORE 1.0 – Predicting Employee Attrition Using a Stacked ML Model

PORE (Predicting Organization Retention & Efficiency) is a machine-learning powered employee attrition prediction system.  
It uses a stacked ensemble model combining multiple algorithms for higher accuracy.  
This repository includes the trained model, preprocessing pipeline, and a full Streamlit web app.

---

## 🚀 Key Features
- Predicts whether an employee is likely to leave the organization  
- Stacked ensemble (Random Forest + XGBoost + Gradient Boosting)  
- Logistic Regression as meta-learner  
- CSV upload or manual input  
- Auto-scaled and encoded preprocessing pipeline  
- Streamlit UI  
- Ready for deployment  

---

## 🧠 Model Architecture

### Base Models
- Random Forest Classifier  
- XGBoost Classifier  
- Gradient Boosting Classifier  

### Meta Model
- Logistic Regression

### Preprocessing
- StandardScaler for numerical features  
- One-Hot Encoding for categorical features  
- train_columns.pkl ensures column order during prediction  

---

## 📁 Repository Structure

PORE-Model/
│
├── models/
│   ├── stacked_attrition_model.pkl
│   ├── scaler.pkl
│   ├── train_columns.pkl
│
├── app/
│   ├── streamlit_app.py
│
├── data/
│   └── sample_input.csv
│
└── README.md

---

## ⚙️ How to Run Locally

### 1. Clone Repository
git clone https://github.com/your-username/PORE-Model.git  
cd PORE-Model

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Streamlit App
streamlit run streamlit_app.py

---

## 📤 Using the App

### A. Upload CSV
- Upload employee dataset  
- Auto preprocessing + prediction + probability
  
---

## 📊 Model Output
- Attrition Prediction: Yes / No  
- Probability Score  
- Key Influencing Features  

---

## 🧪 Model Training Details
- Cleaned dataset  
- Missing value handling  
- Scaling + Encoding  
- Train/Test split 80/20  

### Evaluation Metrics (example)
| Metric | Score |
|--------|--------|
| Accuracy | 0.87 |
| ROC-AUC | 0.92 |
| F1-Score | 0.84 |

(Replace with your actual metrics)

---

## 🌐 Deployment Options
- Streamlit Cloud  
- Render  
- HuggingFace Spaces  
- AWS EC2  

---

## 📦 Requirements
streamlit  
pandas  
numpy  
scikit-learn  
xgboost  
joblib  

---

## 🤝 Contributing
Issues and pull requests are welcome.

---

## 📜 License
MIT License
