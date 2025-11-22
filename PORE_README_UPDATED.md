🧠 PORE 1.0 - Predictive Organizational Retention Engine

PORE (Predictive Organizational Retention Efficiency) is an AI-powered Machine Learning system designed to predict employee attrition.
It helps HR departments identify who is likely to leave and why—leading to better retention strategies and informed decision-making.

🚀 Features

Predicts employee attrition with high accuracy

Uses an advanced stacked ensemble ML model

Provides explainability insights

Automatically preprocesses uploaded datasets

Clean, interactive Streamlit dashboard

🧩 Tech Stack

Python (Pandas, NumPy, Scikit-learn)

Streamlit

Git LFS (for large model files)

Pickle (model serialization)

gdown (auto-downloads model from Google Drive)

📦 Model Files

stacked_attrition_model.pkl

scaler.pkl

num_cols.pkl

train_columns.pkl

stacked_attrition_model.pkl file is automatically downloaded from Google Drive when the app starts.


📘 What’s Included in the PORE Streamlit App
🟦 1. Employee Attrition Prediction

Upload a CSV → app handles preprocessing → model predicts Stay / Leave.

🟩 2. Attrition Probability Score (FI Score)

Shows the probability of an employee leaving, not just the binary label.

🟧 3. Full Data Preprocessing Pipeline

Handles missing values

Applies saved scaler

Encodes categorical columns

Aligns columns with training schema

🟪 4. Feature Engineering Compatibility

Ensures correct data types, ordering, and formatting before running inference.

🟥 5. Explainability & Feature Importance

Displays which features influence predictions the most.

🟨 6. Upload & Preview System

File uploader

Data preview

Column validation

🟫 7. Automatic Model File Downloading

Model files are downloaded on the fly using Google Drive + gdown.

⚪ 8. Clean UI & Dashboard Experience

Simple, modern, and optimized layout for deployment.

🟫 9. Error Handling

Handles:

Missing columns

Wrong file formats

Failed downloads

📊 PORE Model Performance Summary

The model demonstrates strong performance on the test dataset:

Final F1 Score: 0.85+

Accuracy: 0.85+

Precision: 0.85+

Recall: 0.85+

Test Set Evaluation Summary

Balanced classification performance

Strong predictive power

Minimal overfitting

Reliable results across unseen employee profiles

▶️ How to Run the PORE Model Locally
✅ 1. Clone the Repository
git clone https://github.com/arif7866-coderr/PORE-1.0.git
cd PORE-1.0

✅ 2. Create a Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

✅ 3. Install Requirements
pip install -r requirements.txt


Ensure gdown is included:

gdown

✅ 4. Auto-Download Model File

Your script uses:

https://drive.google.com/uc?id=FILE_ID

Example snippet:
import gdown
import os

def download_if_missing(url, out_path):
    if not os.path.exists(out_path):
        gdown.download(url, out_path, quiet=False)

Or you can download Stacked attrition model from this https://drive.google.com/file/d/14BxckoIrTLHYS6woIVwMYYjWSpj8JqYJ/view?usp=drive_link 
after download move it to models folder.

✅ 5. Run the App
streamlit run pore.py

🖥️ Access the App

Open in your browser:

http://localhost:8501

🧑‍💻 Author

Arif Ansari
GitHub: (https://github.com/arif7866-coderr/PORE-1.0)
LinkedIn: https://www.linkedin.com/in/arif-ansari-2b8755391/