# EMPLOYEE PERFORMANCE & RETENTION ANALYSIS
# Complete code for VS Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("EMPLOYEE PERFORMANCE & RETENTION ANALYSIS")
print("="*60)

# Load the dataset
print("Loading dataset...")
url = "https://www.dropbox.com/scl/fi/huvcvi6vv5w986a0n3znp/Employee_Performance_Retention.csv?rlkey=vhd2bpbjrxzakkdi0q3pyaygz&e=1&st=gvsv4dza&dl=1"
df = pd.read_csv(url)

print("✓ Dataset loaded successfully!")
print(f"✓ Dataset shape: {df.shape}")

# Data Exploration
print("\n" + "="*60)
print("DATA EXPLORATION")
print("="*60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

print("\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Check the target variable distribution
print(f"\nAttrition distribution:")
print(df['Attrition'].value_counts())
print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean():.2%}")

print(f"\nPerformance Rating distribution:")
print(df['Performance Rating'].value_counts().sort_index())

# Data Preprocessing Functions
def prepare_data_classification(df, target_col='Attrition'):
    """Prepare data for classification tasks"""
    data = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Department', 'Job Satisfaction Level', 'Promotion in Last 2 Years']
    
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    # Prepare features and target
    X = data.drop(['Employee ID', target_col, 'Performance Rating'], axis=1)
    y = le.fit_transform(data[target_col])  # Yes=1, No=0
    
    return X, y, X.columns.tolist()

def prepare_data_regression(df, target_col='Performance Rating'):
    """Prepare data for regression tasks"""
    data = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Department', 'Job Satisfaction Level', 'Promotion in Last 2 Years', 'Attrition']
    
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    # Prepare features and target
    X = data.drop(['Employee ID', target_col], axis=1)
    y = data[target_col]
    
    return X, y

# Task 1: Random Forest
print("\n" + "="*60)
print("TASK 1: RANDOM FOREST")
print("="*60)

# Classification - Predict Attrition
print("\n--- Random Forest Classification (Attrition Prediction) ---")

X_class, y_class, feature_names = prepare_data_classification(df)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
)

print(f"Training set size: {X_train_class.shape[0]}")
print(f"Test set size: {X_test_class.shape[0]}")
print(f"Features: {feature_names}")

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_classifier.fit(X_train_class, y_train_class)

# Predictions
y_pred_rf = rf_classifier.predict(X_test_class)

# Evaluation
print("\n📊 Random Forest Classification Results:")
print(f"Accuracy: {accuracy_score(y_test_class, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test_class, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test_class, y_pred_rf):.4f}")
print(f"F1-Score: {2 * (precision_score(y_test_class, y_pred_rf) * recall_score(y_test_class, y_pred_rf)) / (precision_score(y_test_class, y_pred_rf) + recall_score(y_test_class, y_pred_rf)):.4f}")

print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_rf, target_names=['No Attrition', 'Attrition']))

# Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Feature Importance Analysis:")
print(feature_importance)

# Regression - Predict Performance Rating
print("\n--- Random Forest Regression (Performance Rating Prediction) ---")

X_reg, y_reg = prepare_data_regression(df)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_regressor.fit(X_train_reg, y_train_reg)

# Predictions and evaluation
y_pred_reg = rf_regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"📊 Regression Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {np.mean(np.abs(y_test_reg - y_pred_reg)):.4f}")

# Task 2: Support Vector Machine (SVM)
print("\n" + "="*60)
print("TASK 2: SUPPORT VECTOR MACHINE")
print("="*60)

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_class)
X_test_scaled = scaler.transform(X_test_class)

# Define different kernels to experiment with
kernels = ['linear', 'poly', 'rbf']
svm_results = {}

print("Training SVM with different kernels...")

for kernel in kernels:
    print(f"\n🔧 Training SVM with {kernel} kernel...")
    start_time = time.time()
    
    if kernel == 'poly':
        svm = SVC(kernel=kernel, degree=3, random_state=42, probability=True)
    else:
        svm = SVC(kernel=kernel, random_state=42, probability=True)
    
    svm.fit(X_train_scaled, y_train_class)
    y_pred_svm = svm.predict(X_test_scaled)
    training_time = time.time() - start_time
    
    # Store results
    svm_results[kernel] = {
        'accuracy': accuracy_score(y_test_class, y_pred_svm),
        'precision': precision_score(y_test_class, y_pred_svm),
        'recall': recall_score(y_test_class, y_pred_svm),
        'training_time': training_time,
        'model': svm
    }
    
    print(f"✅ {kernel.upper()} Kernel Results:")
    print(f"   Accuracy: {svm_results[kernel]['accuracy']:.4f}")
    print(f"   Precision: {svm_results[kernel]['precision']:.4f}")
    print(f"   Recall: {svm_results[kernel]['recall']:.4f}")
    print(f"   Training Time: {training_time:.4f} seconds")

# Compare SVM kernels
print("\n" + "="*40)
print("SVM KERNEL COMPARISON")
print("="*40)

comparison_df = pd.DataFrame(svm_results).T
print(comparison_df[['accuracy', 'precision', 'recall', 'training_time']])

# Task 3: XGBoost
print("\n" + "="*60)
print("TASK 3: XGBOOST")
print("="*60)

# Train basic XGBoost model
print("Training basic XGBoost model...")
start_time = time.time()
xgb_basic = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_basic.fit(X_train_class, y_train_class)
basic_training_time = time.time() - start_time

y_pred_xgb_basic = xgb_basic.predict(X_test_class)

print("📊 Basic XGBoost Results:")
print(f"Accuracy: {accuracy_score(y_test_class, y_pred_xgb_basic):.4f}")
print(f"Precision: {precision_score(y_test_class, y_pred_xgb_basic):.4f}")
print(f"Recall: {recall_score(y_test_class, y_pred_xgb_basic):.4f}")
print(f"Training Time: {basic_training_time:.4f} seconds")

# Hyperparameter Tuning
print("\n🎯 Performing Hyperparameter Tuning...")

# Define parameter grid (small for demonstration)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 0.9]
}

start_time = time.time()
xgb_tuned = XGBClassifier(random_state=42, eval_metric='logloss')
grid_search = GridSearchCV(xgb_tuned, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_class, y_train_class)
tuning_time = time.time() - start_time

print(f"✅ Hyperparameter tuning completed in {tuning_time:.2f} seconds")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train with best parameters
best_xgb = grid_search.best_estimator_
y_pred_xgb_tuned = best_xgb.predict(X_test_class)

print("\n📊 Tuned XGBoost Results:")
print(f"Accuracy: {accuracy_score(y_test_class, y_pred_xgb_tuned):.4f}")
print(f"Precision: {precision_score(y_test_class, y_pred_xgb_tuned):.4f}")
print(f"Recall: {recall_score(y_test_class, y_pred_xgb_tuned):.4f}")

# XGBoost Feature Importance
xgb_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 XGBoost Feature Importance:")
print(xgb_importance)

# Final Model Comparison
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)

# Collect all results
models_comparison = {
    'Random Forest': {
        'accuracy': accuracy_score(y_test_class, y_pred_rf),
        'precision': precision_score(y_test_class, y_pred_rf),
        'recall': recall_score(y_test_class, y_pred_rf),
        'training_time': 0.1  # approximate
    },
    'SVM (RBF)': svm_results['rbf'],
    'XGBoost (Basic)': {
        'accuracy': accuracy_score(y_test_class, y_pred_xgb_basic),
        'precision': precision_score(y_test_class, y_pred_xgb_basic),
        'recall': recall_score(y_test_class, y_pred_xgb_basic),
        'training_time': basic_training_time
    },
    'XGBoost (Tuned)': {
        'accuracy': accuracy_score(y_test_class, y_pred_xgb_tuned),
        'precision': precision_score(y_test_class, y_pred_xgb_tuned),
        'recall': recall_score(y_test_class, y_pred_xgb_tuned),
        'training_time': tuning_time + basic_training_time
    }
}

# Create comparison dataframe
comparison_df_final = pd.DataFrame(models_comparison).T
print("\n📈 Model Performance Comparison:")
print(comparison_df_final)

# Visualization Section
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Create a comprehensive visualization dashboard
plt.figure(figsize=(20, 15))

# 1. Dataset Overview
plt.subplot(3, 4, 1)
df['Attrition'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title('Attrition Distribution')

plt.subplot(3, 4, 2)
df['Performance Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Performance Rating Distribution')
plt.xlabel('Performance Rating')
plt.ylabel('Count')

plt.subplot(3, 4, 3)
df['Department'].value_counts().plot(kind='bar')
plt.title('Department Distribution')
plt.xticks(rotation=45)

plt.subplot(3, 4, 4)
df['Job Satisfaction Level'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Job Satisfaction Distribution')

# 2. Feature Importance Comparison
plt.subplot(3, 4, 5)
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Random Forest - Top 10 Features')
plt.xlabel('Importance')

plt.subplot(3, 4, 6)
sns.barplot(data=xgb_importance.head(10), x='importance', y='feature')
plt.title('XGBoost - Top 10 Features')
plt.xlabel('Importance')

# 3. Model Performance Comparison
plt.subplot(3, 4, 7)
metrics_plot = comparison_df_final[['accuracy', 'precision', 'recall']]
metrics_plot.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(3, 4, 8)
comparison_df_final['training_time'].plot(kind='bar', color='orange')
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)

# 4. Confusion Matrices
plt.subplot(3, 4, 9)
cm_rf = confusion_matrix(y_test_class, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('RF Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(3, 4, 10)
cm_xgb = confusion_matrix(y_test_class, y_pred_xgb_tuned)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens')
plt.title('XGBoost Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(3, 4, 11)
best_svm_kernel = max(svm_results, key=lambda k: svm_results[k]['accuracy'])
cm_svm = confusion_matrix(y_test_class, svm_results[best_svm_kernel]['model'].predict(X_test_scaled))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds')
plt.title(f'SVM ({best_svm_kernel}) Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()

# Summary and Recommendations
print("\n" + "="*60)
print("SUMMARY AND RECOMMENDATIONS")
print("="*60)

# Determine best model
best_model_name = comparison_df_final['accuracy'].idxmax()
best_accuracy = comparison_df_final.loc[best_model_name, 'accuracy']

print(f"🏆 BEST PERFORMING MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")

print("\n🔑 KEY FINDINGS:")
print(f"1. Top 3 factors affecting attrition: {list(feature_importance.head(3)['feature'].values)}")
print(f"2. Dataset attrition rate: {(df['Attrition'] == 'Yes').mean():.2%}")
print(f"3. Most important feature: {feature_importance.iloc[0]['feature']}")

print("\n💡 BUSINESS RECOMMENDATIONS:")
print("1. Focus on improving job satisfaction to reduce attrition")
print("2. Monitor working hours and prevent burnout")
print("3. Provide regular training and development opportunities")
print("4. Implement fair promotion policies")
print("5. Use the model to identify at-risk employees for proactive retention")

print("\n🤖 MODEL SELECTION GUIDANCE:")
print(f"• For production: {best_model_name} (highest accuracy)")
print("• For interpretability: Random Forest")
print("• For speed: Basic XGBoost")
print("• For linear relationships: SVM with linear kernel")

print("\n" + "="*60)
print("ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
print("="*60)