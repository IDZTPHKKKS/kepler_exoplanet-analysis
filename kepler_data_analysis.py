import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('kepler.csv')

# Inspect the dataset
print(data.head())
print(data.info())

# Encode the target variable
label_encoder = LabelEncoder()
data['koi_disposition'] = label_encoder.fit_transform(data['koi_disposition'])

# Handle missing values
# Drop columns with too many missing values (example threshold: 50%)
threshold = 0.5
data = data.loc[:, data.isnull().mean() < threshold]

# Select only numeric columns for median imputation
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Impute missing values for numeric columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Select relevant features and target
X = data[['koi_period', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_model_snr']]  # Features
y = data['koi_disposition']  # Target

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Optimize hyperparameters with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV with cross-validation and parallel processing
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
