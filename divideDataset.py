import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import json

# Load data from JSON file
with open('./out_data/dataset.json', 'r') as file:
    data = json.load(file)

# Convert data to DataFrame
df = pd.DataFrame(data)

# One-hot encode categorical columns
categorical_columns = ['a46', 'a47']  # Update with the actual names of your categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Separate features and target variable
X = df_encoded.drop(['Class'], axis=1)
y = df_encoded['Class']

# Method 1: Information gain attributes selection
selector_info_gain = SelectKBest(mutual_info_classif, k=5)
X_info_gain = selector_info_gain.fit_transform(X, y)
selected_features_info_gain = X.columns[selector_info_gain.get_support()]
df_info_gain = pd.concat([df_encoded['Class'], df_encoded[selected_features_info_gain]], axis=1)
df_info_gain.to_csv('info_gain_selected_attributes.csv', index=False)

# Method 2: Correlation method
correlation_matrix = df_encoded.corr()
correlation_threshold = 0.5
correlated_columns = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            correlated_columns.append(colname)
df_corr = df_encoded.drop(correlated_columns, axis=1)
df_corr.to_csv('correlation_selected_attributes.csv', index=False)

# Method 3: Recursive Feature Elimination (RFE) with Random Forest
estimator = RandomForestClassifier()  # You can replace this with any other estimator suitable for your problem
selector_rfe = RFE(estimator, n_features_to_select=5, step=1)
X_rfe = selector_rfe.fit_transform(X, y)
selected_features_rfe = X.columns[selector_rfe.get_support()]
df_rfe = pd.concat([df_encoded['Class'], df_encoded[selected_features_rfe]], axis=1)
df_rfe.to_csv('rfe_selected_attributes.csv', index=False)
