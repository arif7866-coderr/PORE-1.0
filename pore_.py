# ==================================================
# PORE Attrition Prediction Model
# Stacked Random Forest + XGBoost
# With SMOTE, Complete Plots & Saved Columns
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import shap

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
df = pd.read_csv(r"C:\Models\PORE 1.0\Data\PORE.csv")
df = df.drop(columns=['employee_id'], errors='ignore')

# -----------------------------
# 2️⃣ Preprocessing
# -----------------------------
target_col = 'attrition'
y = df[target_col]
X = df.drop(columns=[target_col])

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Save column info for test alignment
joblib.dump(X.columns, "train_columns.pkl")
joblib.dump(num_cols, "num_cols.pkl")

# -----------------------------
# 3️⃣ Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4️⃣ Apply SMOTE to Training Data
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train) #type: ignore

# -----------------------------
# 5️⃣ Base Models + Hyperparameter Tuning
# -----------------------------
# Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train_res, y_train_res) # type: ignore
best_rf = rf_grid.best_estimator_

# XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 1]
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train_res, y_train_res) # type: ignore
best_xgb = xgb_grid.best_estimator_

# -----------------------------
# 6️⃣ Stacking Model
# -----------------------------
estimators = [('rf', best_rf), ('xgb', best_xgb)]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=5
)
stack_model.fit(X_train_res, y_train_res)

# -----------------------------
# 7️⃣ Model Evaluation
# -----------------------------
y_pred = stack_model.predict(X_test)
y_pred_prob = stack_model.predict_proba(X_test)[:,1] # type: ignore

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 7.1 Confusion Matrix Plot
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Attrition','Attrition'], yticklabels=['No Attrition','Attrition'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# 8️⃣ Save Model & Preprocessors
# -----------------------------
joblib.dump(stack_model, "stacked_attrition_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 9️⃣ Plots
# -----------------------------
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = (10,6)

# 9.1 Target Distribution
plt.figure()
sns.countplot(x=y, palette='viridis')
plt.title('Attrition Distribution')
plt.show()

# 9.2 Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(pd.concat([X, y], axis=1).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 9.3 Feature Importance (Random Forest)
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# 9.4 Boxplots for Key Features
important_features = ['age', 'tenure_months', 'salary_band', 'avg_monthly_hours']
for feature in important_features:
    if feature in X.columns:
        plt.figure()
        sns.boxplot(x=y, y=X[feature], palette='Set2')
        plt.title(f'{feature} vs Attrition')
        plt.show()

# 9.5 SHAP Summary Plot
explainer = shap.TreeExplainer(stack_model.estimators_[0][1])  # type: ignore
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

# 9.6 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
