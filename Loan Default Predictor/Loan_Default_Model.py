import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')


print("Loading dataset...")
df = pd.read_csv('Loan_Default.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nTarget variable (Status) distribution:")
print(df['Status'].value_counts())
print("\nTarget variable unique values:")
print(df['Status'].unique())
print(f"\nMissing values in Status: {df['Status'].isnull().sum()}")


df = df.dropna(subset=['Status'])


if df['Status'].dtype == 'object':
    print("\nConverting Status to numeric...")
    status_mapping = {'0': 0, '1': 1, 0: 0, 1: 1}
    df['Status'] = df['Status'].map(status_mapping)
    df = df.dropna(subset=['Status'])

print(f"\nFinal Status distribution:")
print(df['Status'].value_counts())


selected_columns = [
    'loan_amount', 'rate_of_interest', 'term', 'property_value', 
    'income', 'Gender', 'Credit_Worthiness', 'business_or_commercial', 'age'
]


available_columns = [col for col in selected_columns if col in df.columns]
print(f"\nUsing columns: {available_columns}")


X = df[available_columns].copy()
y = df['Status'].astype(int)


if 'Credit_Worthiness' in X.columns:
    print("\nCredit_Worthiness vs Status:")
    credit_status = df.groupby('Credit_Worthiness')['Status'].mean()
    print(credit_status)
    
    credit_mapping = {'l1': 'poor' if credit_status.get('l1', 0) > 0.5 else 'good'}
    X['Credit_Worthiness'] = X['Credit_Worthiness'].astype(str).map(credit_mapping).fillna('good')
    df['Credit_Worthiness'] = df['Credit_Worthiness'].astype(str).map(credit_mapping).fillna('good')

for col in ['Gender', 'business_or_commercial']:
    if col in X.columns:
        X[col] = X[col].astype(str)
        df[col] = df[col].astype(str)


X['LTV'] = (X['loan_amount'] / X['property_value'] * 100).clip(upper=1000)
X['DTI'] = (X['loan_amount'] / X['term'] / X['income'] * 100).clip(upper=1000)
df['LTV'] = (df['loan_amount'] / df['property_value'] * 100).clip(upper=1000)
df['DTI'] = (df['loan_amount'] / df['term'] / df['income'] * 100).clip(upper=1000)


available_columns.extend(['LTV', 'DTI'])
print(f"\nUpdated columns with LTV and DTI: {available_columns}")

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

print(f"\nMissing values per column:")
print(X.isnull().sum())


missing_threshold = 0.5 * len(available_columns)
X = X.dropna(thresh=len(available_columns) - missing_threshold)
y = y.loc[X.index]

print(f"\nAfter cleaning - Feature matrix shape: {X.shape}")


if 'age' in X.columns and X['age'].dtype == 'object':
    print("\nConverting categorical age to numeric...")
    age_mapping = {
        '25-34': 29.5, '35-44': 39.5, '45-54': 49.5, 
        '55-64': 59.5, '65-74': 69.5, '<25': 22, '>74': 77
    }
    X['age'] = X['age'].map(age_mapping)
    df['age'] = df['age'].map(age_mapping)

print("\nAdding synthetic high-risk default cases...")
synthetic_rows = pd.DataFrame({
    'loan_amount': [600000] * 200,
    'rate_of_interest': [12.0] * 200,
    'term': [360] * 200,
    'property_value': [100000] * 200,
    'income': [300] * 200,
    'Gender': ['Sex Not Available'] * 200,
    'Credit_Worthiness': ['poor'] * 200,
    'business_or_commercial': ['b/c'] * 200,
    'age': [22] * 200,
    'LTV': [600.0] * 200,
    'DTI': [555.6] * 200,
    'Status': [1] * 200
})
df = pd.concat([df, synthetic_rows], ignore_index=True)


X = df[available_columns].copy()
y = df['Status'].astype(int)


for col in ['Gender', 'Credit_Worthiness', 'business_or_commercial']:
    if col in X.columns:
        X[col] = X[col].astype(str)


print("\nUnique values in categorical columns:")
for col in ['Gender', 'Credit_Worthiness', 'business_or_commercial']:
    if col in X.columns:
        print(f"{col}: {X[col].unique()}")


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])


preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

print(f"\nClass distribution before oversampling:")
print(f"No Default (0): {(y == 0).sum()} ({(y == 0).mean():.2%})")
print(f"Default (1): {(y == 1).sum()} ({(y == 1).mean():.2%})")


if (y == 1).mean() < 0.5:
    print("\nDataset is imbalanced. Oversampling defaults to 1:1 ratio...")
    df_balanced = X.copy()
    df_balanced['Status'] = y
    
    df_majority = df_balanced[df_balanced['Status'] == 0]
    df_minority = df_balanced[df_balanced['Status'] == 1]
    
    n_samples = len(df_majority)
    df_minority_oversampled = df_minority.sample(n=n_samples, replace=True, random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_oversampled], ignore_index=True)
    
    X = df_balanced[available_columns].copy()
    y = df_balanced['Status'].astype(int)
    
    print(f"After oversampling:")
    print(f"No Default (0): {(y == 0).sum()} ({(y == 0).mean():.2%})")
    print(f"Default (1): {(y == 1).sum()} ({(y == 1).mean():.2%})")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train default rate: {y_train.mean():.2%}")
print(f"Test default rate: {y_test.mean():.2%}")


models = {
    'Random Forest': RandomForestClassifier(
        random_state=42, n_estimators=300, max_depth=25, 
        min_samples_split=3, min_samples_leaf=1, class_weight='balanced_subsample'
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42, max_depth=10, class_weight='balanced'
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced'
    )
}

best_model = None
best_accuracy = 0
best_model_name = ''

print(f"\n{'='*60}")
for name, classifier in models.items():
    print(f"Training {name}...")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"\n{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nSample predictions (first 10):")
    sample_proba = y_pred_proba[:10]
    for i, (pred, proba) in enumerate(zip(y_pred[:10], sample_proba)):
        print(f"Sample {i+1}: Prediction={pred}, Probability=[{proba[0]:.3f}, {proba[1]:.3f}]")
    
    threshold = 0.25
    y_pred_adjusted = (y_pred_proba[:, 1] >= threshold).astype(int)
    print(f"\nClassification Report (Threshold={threshold}):")
    print(classification_report(y_test, y_pred_adjusted))
    print(f"\nConfusion Matrix (Threshold={threshold}):")
    print(confusion_matrix(y_test, y_pred_adjusted))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name

joblib.dump(best_model, 'loan_default_model.pkl')
print(f"\nBest model ({best_model_name}, Accuracy: {best_accuracy:.4f}) saved as 'loan_default_model.pkl'")

feature_info = {
    'selected_columns': available_columns,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'target_distribution': y.value_counts().to_dict()
}
joblib.dump(feature_info, 'feature_info.pkl')
print("Feature info saved as 'feature_info.pkl'")

print(f"\n{'='*60}")
print("TESTING EXTREME CASES WITH BEST MODEL:")

test_cases = []

high_risk = {col: X[col].median() if col in numeric_features else X[col].mode()[0] 
             for col in available_columns}
high_risk.update({
    'loan_amount': 600000,
    'rate_of_interest': 12.0,
    'term': 360,
    'property_value': 100000,
    'income': 300,
    'Gender': 'Sex Not Available',
    'Credit_Worthiness': 'poor',
    'business_or_commercial': 'b/c',
    'age': 22,
    'LTV': 600.0,
    'DTI': 555.6
})
test_cases.append(("High Risk", high_risk))

low_risk = {col: X[col].median() if col in numeric_features else X[col].mode()[0] 
            for col in available_columns}
low_risk.update({
    'loan_amount': 150000,
    'rate_of_interest': 3.0,
    'term': 360,
    'property_value': 300000,
    'income': 8000,
    'Gender': 'Male',
    'Credit_Worthiness': 'good',
    'business_or_commercial': 'nob/c',
    'age': 45,
    'LTV': 50.0,
    'DTI': 5.2
})
test_cases.append(("Low Risk", low_risk))

for case_name, case_data in test_cases:
    test_df = pd.DataFrame([case_data])
    pred = best_model.predict(test_df)[0]
    proba = best_model.predict_proba(test_df)[0]
    
    print(f"\n{case_name} Case:")
    print(f"Input: {case_data}")
    print(f"Prediction: {pred} (0=No Default, 1=Default)")
    print(f"Probabilities: No Default={proba[0]:.3f}, Default={proba[1]:.3f}")
    print(f"LTV: {case_data['LTV']:.1f}%")
    print(f"DTI: {case_data['DTI']:.1f}%")

print("\nModel training completed!")