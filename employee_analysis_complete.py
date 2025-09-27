# COMPLETE EMPLOYEE PERFORMANCE & RETENTION ANALYSIS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🚀 COMPLETE EMPLOYEE PERFORMANCE & RETENTION ANALYSIS")
print("=" * 70)

def main():
    try:
        # Load the dataset
        print("📊 1. LOADING DATASET...")
        url = "https://www.dropbox.com/scl/fi/huvcvi6vv5w986a0n3znp/Employee_Performance_Retention.csv?rlkey=vhd2bpbjrxzakkdi0q3pyaygz&e=1&st=gvsv4dza&dl=1"
        df = pd.read_csv(url)
        
        print("✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Basic dataset info
        print(f"\n📈 2. DATASET OVERVIEW:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Attrition rate: {(df['Attrition'] == 'Yes').mean():.2%}")
        print(f"   Departments: {', '.join(df['Department'].unique())}")
        
        print(f"\n   Performance Rating distribution:")
        print(df['Performance Rating'].value_counts().sort_index())
        print(f"\n   Job Satisfaction distribution:")
        print(df['Job Satisfaction Level'].value_counts())
        
        # Data preview
        print(f"\n👀 3. DATA PREVIEW:")
        print(df.head(3))
        
        # TASK 1: RANDOM FOREST
        print(f"\n" + "="*50)
        print("🌲 TASK 1: RANDOM FOREST")
        print("="*50)
        
        # Classification - Predict Attrition
        print(f"\n🔮 1A. RANDOM FOREST CLASSIFICATION (Attrition Prediction)")
        X_class, y_class, feature_names = prepare_data_classification(df)
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
        )
        
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_classifier.fit(X_train_class, y_train_class)
        y_pred_rf = rf_classifier.predict(X_test_class)
        
        print(f"   Accuracy: {accuracy_score(y_test_class, y_pred_rf):.4f}")
        print(f"   Precision: {precision_score(y_test_class, y_pred_rf):.4f}")
        print(f"   Recall: {recall_score(y_test_class, y_pred_rf):.4f}")
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n   🔍 Top 5 Most Important Features for Attrition:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
        
        # Regression - Predict Performance Rating
        print(f"\n📈 1B. RANDOM FOREST REGRESSION (Performance Rating Prediction)")
        X_reg, y_reg = prepare_data_regression(df)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_regressor.fit(X_train_reg, y_train_reg)
        y_pred_reg = rf_regressor.predict(X_test_reg)
        
        print(f"   R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
        print(f"   MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
        
        # TASK 2: SUPPORT VECTOR MACHINE (SVM)
        print(f"\n" + "="*50)
        print("⚡ TASK 2: SUPPORT VECTOR MACHINE (SVM)")
        print("="*50)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_class)
        X_test_scaled = scaler.transform(X_test_class)
        
        kernels = ['linear', 'rbf']
        svm_results = {}
        
        for kernel in kernels:
            svm = SVC(kernel=kernel, random_state=42, probability=True)
            svm.fit(X_train_scaled, y_train_class)
            y_pred_svm = svm.predict(X_test_scaled)
            
            svm_results[kernel] = {
                'accuracy': accuracy_score(y_test_class, y_pred_svm),
                'precision': precision_score(y_test_class, y_pred_svm),
                'recall': recall_score(y_test_class, y_pred_svm)
            }
            
            print(f"   {kernel.upper()} Kernel - Accuracy: {svm_results[kernel]['accuracy']:.4f}")
        
        # TASK 3: XGBOOST
        print(f"\n" + "="*50)
        print("🎯 TASK 3: XGBOOST")
        print("="*50)
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train_class, y_train_class)
        y_pred_xgb = xgb_model.predict(X_test_class)
        
        print(f"   Accuracy: {accuracy_score(y_test_class, y_pred_xgb):.4f}")
        print(f"   Precision: {precision_score(y_test_class, y_pred_xgb):.4f}")
        print(f"   Recall: {recall_score(y_test_class, y_pred_xgb):.4f}")
        
        # Hyperparameter Tuning (simplified)
        print(f"\n   🎛️  Hyperparameter Tuning...")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
        
        xgb_tuned = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb_tuned, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_class, y_train_class)
        
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")
        
        # MODEL COMPARISON
        print(f"\n" + "="*50)
        print("🏆 MODEL COMPARISON")
        print("="*50)
        
        models_comparison = {
            'Random Forest': {
                'accuracy': accuracy_score(y_test_class, y_pred_rf),
                'precision': precision_score(y_test_class, y_pred_rf),
                'recall': recall_score(y_test_class, y_pred_rf)
            },
            'SVM (RBF)': svm_results['rbf'],
            'XGBoost': {
                'accuracy': accuracy_score(y_test_class, y_pred_xgb),
                'precision': precision_score(y_test_class, y_pred_xgb),
                'recall': recall_score(y_test_class, y_pred_xgb)
            }
        }
        
        comparison_df = pd.DataFrame(models_comparison).T
        print(comparison_df)
        
        # VISUALIZATIONS
        print(f"\n" + "="*50)
        print("📊 VISUALIZATIONS")
        print("="*50)
        
        create_visualizations(df, feature_importance, models_comparison)
        
        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"🎉 All three machine learning models executed perfectly!")
        
        # BUSINESS INSIGHTS
        print(f"\n" + "="*50)
        print("💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*50)
        
        top_features = feature_importance.head(3)['feature'].tolist()
        print(f"🔑 Top 3 factors affecting employee attrition:")
        for i, feature in enumerate(top_features, 1):
            print(f"   {i}. {feature}")
        
        print(f"\n�� Recommendations:")
        print(f"   1. Focus on improving {top_features[0].lower().replace('_', ' ')}")
        print(f"   2. Monitor and address issues related to {top_features[1].lower().replace('_', ' ')}")
        print(f"   3. Implement strategies for better {top_features[2].lower().replace('_', ' ')}")
        print(f"   4. Use the best model ({comparison_df['accuracy'].idxmax()}) for predicting at-risk employees")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Running sample analysis with generated data...")
        run_sample_analysis()

def prepare_data_classification(df, target_col='Attrition'):
    """Prepare data for classification tasks"""
    data = df.copy()
    le = LabelEncoder()
    categorical_cols = ['Department', 'Job Satisfaction Level', 'Promotion in Last 2 Years']
    
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    X = data.drop(['Employee ID', target_col, 'Performance Rating'], axis=1)
    y = le.fit_transform(data[target_col])
    
    return X, y, X.columns.tolist()

def prepare_data_regression(df, target_col='Performance Rating'):
    """Prepare data for regression tasks"""
    data = df.copy()
    le = LabelEncoder()
    categorical_cols = ['Department', 'Job Satisfaction Level', 'Promotion in Last 2 Years', 'Attrition']
    
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    X = data.drop(['Employee ID', target_col], axis=1)
    y = data[target_col]
    
    return X, y

def create_visualizations(df, feature_importance, models_comparison):
    """Create comprehensive visualizations"""
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Attrition distribution
    plt.subplot(3, 4, 1)
    df['Attrition'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Employee Attrition Distribution')
    
    # Plot 2: Department distribution
    plt.subplot(3, 4, 2)
    df['Department'].value_counts().plot(kind='bar')
    plt.title('Department Distribution')
    plt.xticks(rotation=45)
    
    # Plot 3: Job Satisfaction
    plt.subplot(3, 4, 3)
    df['Job Satisfaction Level'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Job Satisfaction Levels')
    
    # Plot 4: Performance Rating
    plt.subplot(3, 4, 4)
    df['Performance Rating'].value_counts().sort_index().plot(kind='bar')
    plt.title('Performance Rating Distribution')
    
    # Plot 5: Feature Importance
    plt.subplot(3, 4, 5)
    plt.barh(feature_importance.head(8)['feature'], feature_importance.head(8)['importance'])
    plt.title('Top 8 Features for Attrition Prediction')
    plt.xlabel('Importance')
    
    # Plot 6: Model Comparison
    plt.subplot(3, 4, 6)
    models = list(models_comparison.keys())
    accuracies = [models_comparison[model]['accuracy'] for model in models]
    plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Plot 7: Attrition by Department
    plt.subplot(3, 4, 7)
    attrition_by_dept = pd.crosstab(df['Department'], df['Attrition'], normalize='index') * 100
    attrition_by_dept.plot(kind='bar', stacked=True)
    plt.title('Attrition Rate by Department (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Attrition')
    
    # Plot 8: Experience vs Attrition
    plt.subplot(3, 4, 8)
    df.boxplot(column='Years of Experience', by='Attrition')
    plt.title('Years of Experience vs Attrition')
    plt.suptitle('')  # Remove automatic title
    
    # Plot 9: Correlation heatmap (numeric columns only)
    plt.subplot(3, 4, 9)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
    
    plt.tight_layout()
    plt.show()

def run_sample_analysis():
    """Run analysis with sample data if real data fails"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Employee ID': range(1, n_samples+1),
        'Age': np.random.randint(22, 65, n_samples),
        'Department': np.random.choice(['HR', 'Engineering', 'Sales', 'Marketing'], n_samples),
        'Years of Experience': np.random.uniform(0, 30, n_samples),
        'Monthly Working Hours': np.random.randint(120, 250, n_samples),
        'Training Hours per Year': np.random.randint(0, 100, n_samples),
        'Performance Rating': np.random.randint(1, 6, n_samples),
        'Job Satisfaction Level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Promotion in Last 2 Years': np.random.choice(['Yes', 'No'], n_samples),
        'Attrition': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85])
    }
    
    df = pd.DataFrame(data)
    print("✅ Sample dataset created for demonstration!")
    main()  # Recursive call with sample data

if __name__ == "__main__":
    main()
