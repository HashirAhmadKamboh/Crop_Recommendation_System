# 🌱 Crop Recommendation System

This project is a **machine learning-based Crop Recommendation System** that predicts the most suitable crop to cultivate based on soil and environmental parameters. By analyzing key features like nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall, the system provides actionable crop suggestions to assist farmers in maximizing yield and sustainability.

## 🚀 Project Features

* ✅ End-to-end machine learning pipeline.
* ✅ Data preprocessing including label encoding and feature scaling.
* ✅ Comparative evaluation of **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Random Forest Classifier**.
* ✅ Performance validation using confusion matrices and classification reports.
* ✅ Hyperparameter tuning for improved model accuracy.

## 📂 Dataset

The dataset includes the following features:

* **N**: Nitrogen content in soil
* **P**: Phosphorus content in soil
* **K**: Potassium content in soil
* **Temperature**: In degrees Celsius
* **Humidity**: In percentage
* **pH**: Acidity/alkalinity level of soil
* **Rainfall**: In mm
* **Label**: Recommended crop type

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib & Seaborn (for visualizations)

## 📝 Key Steps

1. **Data Cleaning & Exploration**

   * Checked for missing values.
   * Analyzed feature distributions and relationships.

2. **Data Preprocessing**

   * Label encoding of crop names.
   * Feature scaling using both StandardScaler and MinMaxScaler.

3. **Model Training**

   * K-Nearest Neighbors (KNN)
   * Support Vector Machine (SVM)
   * Random Forest Classifier

4. **Model Evaluation**

   * Accuracy Scores
   * Confusion Matrices
   * Classification Reports

5. **Hyperparameter Tuning**

   * Optimized model settings for best performance.

## ⚙️ Challenges Overcome

* Handled feature scaling discrepancies by creating a multi-step scaling process.
* Performed careful hyperparameter tuning to avoid overfitting.
* Ensured balanced performance across multiple crop classes.

## 📈 Results

All three models performed well, with the Random Forest classifier generally providing the most robust and balanced predictions.

## 🤝 Future Improvements

* Integrate real-time weather APIs for dynamic predictions.
* Build a web or mobile interface for user-friendly crop recommendations.

## 📬 Contact

If you have any questions or suggestions, feel free to reach out:
**Hashir Ahmad**
Email: \hashirahmadkamboh@gmail.com


