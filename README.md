# Breast_cancer_Detection
Machine learning (ML) techniques can help identify breast cancer early
--------------------------------------------------------------------------------------
Project Overview
The Breast Cancer Prediction System is a machine learning-based web application that predicts whether a breast tumor is Malignant (Cancerous) or Benign (Non-Cancerous). The system uses Flask for the web framework

Technologies Used :<br>
Frontend: HTML, CSS
Backend: Python (Flask)
Machine Learning Model: Trained using Scikit-learn
Database: N/A (No database is used as the model loads pre-trained data)
--------------------------------------------------------------------------------------

How It Works
*The user inputs breast cancer-related features (e.g., radius mean, texture mean, etc.).
*The system preprocesses the data using the scaler (scaler.pkl).
*The machine learning model (BCP.pkl) predicts whether the tumor is Malignant (M) or Benign (B).
*The result is displayed on a separate results page.
