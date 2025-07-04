# Liver-Disease-Prediction

[![Language](https://img.shields.io/badge/Language-Jupyter%20Notebook-yellow.svg?style=for-the-badge)](https://en.wikipedia.org/wiki/Programming_language)

This project aims to develop a machine learning model for predicting liver disease. It involves data analysis, model training, and a prediction interface that allows users to interact with the trained model.

---

## 📑 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Features

- **Data Preprocessing:** Handling missing values and categorical variables.
- **Exploratory Data Analysis (EDA):** Visual insights using plots and statistics (e.g., Altair).
- **Model Training:** Includes models like Decision Tree (saved as `dt_model.joblib`) trained on liver disease dataset.
- **Model Persistence:** Uses `joblib` to store and reuse trained models and encoders.
- **Prediction Script:** `app.py` provides a prediction interface, possibly via CLI or a web app.
- **Notebook-driven Development:** Training and analysis steps are included in Jupyter Notebooks (`liver_train.ipynb`, `Liver_Disease_Prediction.ipynb`).

---

## 🧰 Technologies Used

- **Languages/Environments:**
  - Jupyter Notebook
  - Python

- **Key Libraries:**
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `altair`
  - `joblib`
  - ...and others listed in `requirements.txt`

---

## ⚙️ Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/liver-disease-prediction.git
    cd liver-disease-prediction
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Usage

### 🧪 Run the Jupyter Notebook(s)

Use the notebooks for training, testing, and analyzing the dataset:

```bash
jupyter notebook Liver_Disease_Prediction.ipynb
# or
jupyter notebook liver_train.ipynb
```

> These notebooks will guide you through data analysis and model training steps.

### 🤖 Run the Prediction App

Once the model is trained and saved:

```bash
python app.py
```

> This may start a terminal-based input system or a web interface depending on the app's implementation.

Ensure that `dt_model.joblib` and `label_encoders.joblib` are present in the correct directory before running the app.

---

## 🤝 Contributing

We welcome your contributions!

1. Fork the repository
2. Create your feature branch:
    ```bash
    git checkout -b feature/AmazingFeature
    ```
3. Commit your changes:
    ```bash
    git commit -m "Add some AmazingFeature"
    ```
4. Push to your branch:
    ```bash
    git push origin feature/AmazingFeature
    ```
5. Open a pull request

---

