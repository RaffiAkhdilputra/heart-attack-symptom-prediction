# Heart Attack Symptom Prediction

## Project Overview  
This project aims to develop a **machine learning model** capable of predicting the likelihood of a heart attack based on key health parameters and symptoms. It includes data exploration, preprocessing, model training, evaluation, and optional deployment in an interactive notebook (Jupyter/Google Colab).

---

## Features  
- Exploratory Data Analysis (EDA) with visualization of key health metrics  
- Data preprocessing: missing value handling, categorical encoding, and feature scaling  
- Model training using algorithms such as Logistic Regression, Random Forest, or others  
- Model evaluation with metrics like **accuracy**, **confusion matrix**, and **ROC-AUC**  
- Interactive interface for real-time prediction (via notebook or web app prototype)  
- Fully reproducible workflow with documented steps and environment setup  

---

## Project Structure  

    ├── data/ ← dataset or link to data source
    ├── notebooks/ ← Jupyter or Google Colab notebooks
    ├── src/ ← modular Python scripts (if any)
    ├── models/ ← trained models (.pkl, .joblib, etc.)
    ├── requirements.txt ← dependency list
    └── README.md ← project documentation

---

## Installation  
1. Clone the repository:  
    ```bash
    git clone https://github.com/RaffiAkhdilputra/heart-attack-symptom-prediction.git
    cd heart-attack-symptom-prediction

    (Optional) Create a virtual environment:
    
    python3 -m venv venv
    source venv/bin/activate        # on macOS/Linux
    venv\Scripts\activate           # on Windows
    
2. Install dependencies:
    ```python  
    pip install -r requirements.txt
    Run the notebook or script for training and evaluation.

  Usage Example
    
    import pickle
    
    # load model
    model = pickle.load(open('models/model.pkl', 'rb'))
    
    # sample input [age, sex, chest_pain_type, ...]
    sample = [[55, 1, 2, 140, 250, 0, 1, 150, 0, 2.3, 0, 2, 3]]
    prediction = model.predict(sample)
    
    print("Heart Attack Prediction:", "Yes" if prediction[0] == 1 else "No")

---

## Results & Insights

    Achieved model accuracy around XX% (update with actual result).
    Most important features: age, cholesterol, blood pressure, chest pain type, exercise-induced angina.
    Confusion matrix analysis shows strong recall for positive heart attack cases.
    Disclaimer: This model is for educational and research purposes only and should not be used for real medical diagnosis.

---

## Dataset

Heart Attack Prediction
source: https://www.kaggle.com/datasets/imnikhilanand/heart-attack-prediction

---

# License

  UNLICENSE: Feel free to use and modify it for your own purposes.

---

# Contact

  Author: Muhammad Raffi Akhdilputra
  GitHub: @RaffiAkhdilputra
  Email: raffiakdilputra123@gmail.com
