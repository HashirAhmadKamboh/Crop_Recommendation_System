import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
url = "Crop_Recommendation.csv"
df = pd.read_csv(url)

# Encode the categorical 'label' column
le = LabelEncoder()
df['Crop'] = le.fit_transform(df['Crop'])

# Standardize and scale the numerical features
scaler = StandardScaler()
numerical_features = df.columns[:-1]  # All columns except the target
df[numerical_features] = scaler.fit_transform(df[numerical_features])

minmax_scaler = MinMaxScaler()
df[numerical_features] = minmax_scaler.fit_transform(df[numerical_features])

# Split the dataset
X = df.drop('Crop', axis=1)
y = df['Crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train models
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Streamlit dashboard
st.title("Crop Recommendation Dashboard")

# Dataset overview
st.header("Dataset Overview")
st.write(df.head())

# Correlation Matrix
st.header("Correlation Matrix")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Model performance
st.header("Model Performance")

# KNN
st.subheader("K-Nearest Neighbors (KNN)")
st.write(classification_report(y_test, y_pred_knn))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('KNN Confusion Matrix')
st.pyplot(fig)

# SVM
st.subheader("Support Vector Machine (SVM)")
st.write(classification_report(y_test, y_pred_svm))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('SVM Confusion Matrix')
st.pyplot(fig)

# Random Forest
st.subheader("Random Forest (RF)")
st.write(classification_report(y_test, y_pred_rf))
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Random Forest Confusion Matrix')
st.pyplot(fig)

# Run the Streamlit app
# In the terminal, run: streamlit run dashboard.py
