# 🛡️ Phishing Email Detection Model

A complete machine learning project that classifies emails as **Phishing** or **Safe** using Python and Scikit-learn, featuring a beautiful web interface built with Flask.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Future Enhancements](#-future-enhancements)

---

## 🔍 Overview

Phishing emails are one of the most common cybersecurity threats. This project builds a text classification pipeline that detects phishing emails by analyzing their content using:

- **Natural Language Processing (NLP)** for text preprocessing
- **TF-IDF Vectorization** for converting text to numbers
- **Handcrafted Features** (URL count + suspicious keyword count)
- **Two ML Models**: Logistic Regression & Naive Bayes (best is auto-selected)
- **Flask Web Interface** for real-time email classification

---

## 📁 Project Structure

```
phishing/
│
├── dataset_loader.py        # Loads/creates the email dataset
├── preprocessor.py          # Text cleaning and preprocessing
├── feature_extractor.py     # TF-IDF + URL + keyword features
├── model_trainer.py         # Trains LR and NB models
├── evaluator.py             # Metrics + confusion matrix charts
├── predictor.py             # Real-time prediction logic
├── train_and_evaluate.py    # Master script (run this first!)
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
│
├── templates/
│   └── index.html           # Web UI (dark theme, animated)
│
├── models/                  # Auto-created after training
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── best_model.pkl
│   └── feature_extractor.pkl
│
└── reports/                 # Auto-created after evaluation
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_naive_bayes.png
    ├── metrics_logistic_regression.png
    ├── metrics_naive_bayes.png
    └── model_comparison.png
```

---

## ✨ Features

### Data Processing
- ✅ Lowercase conversion
- ✅ Special character removal
- ✅ Stop word removal (NLTK)
- ✅ Tokenization

### Feature Extraction
- ✅ **TF-IDF Vectorization** (top 3000 features, unigrams + bigrams)
- ✅ **URL Detection** — counts suspicious links in email
- ✅ **Suspicious Keyword Detection** — flags words like *urgent*, *verify*, *password*, *bank*

### Machine Learning
- ✅ **Logistic Regression** — linear boundary classifier
- ✅ **Naive Bayes (Multinomial)** — probabilistic text classifier
- ✅ Auto-selects the best performing model

### Evaluation
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Confusion Matrix visualization
- ✅ Model comparison bar chart
- ✅ Charts saved as PNG files in `reports/`

### Web Interface
- ✅ Paste any email and get instant result
- ✅ Shows: Phishing / Safe verdict, confidence %, URLs found, suspicious keywords
- ✅ Load sample phishing or safe email with one click
- ✅ Responsive dark-themed UI with animations

---

## ⚙️ Installation

### 1. Clone or download this project

```bash
cd c:\Users\VELMU\phishing
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the models

```bash
python train_and_evaluate.py
```

This will:
- Load the email dataset
- Preprocess and extract features
- Train both models
- Evaluate and compare them
- Save all models in `models/`
- Save all charts in `reports/`

### Step 2 — Launch the web app

```bash
python app.py
```

Then open your browser and go to:  
**http://127.0.0.1:5000**

### Step 3 — Test individual predictions (optional)

```bash
python predictor.py
```

---

## 🔬 How It Works

```
Raw Email Text
      │
      ▼
 preprocessor.py
  • lowercase
  • remove special chars
  • remove stop words
      │
      ▼
 feature_extractor.py
  • TF-IDF (3000 features)     ──┐
  • URL count                  ──┤──▶ Combined Feature Matrix
  • Suspicious keyword count   ──┘
      │
      ▼
 model_trainer.py
  • Logistic Regression
  • Naive Bayes
  • Best model saved
      │
      ▼
 predictor.py / app.py
  → "Phishing Email" or "Safe Email"
```

---

## 📊 Model Performance

After training on the built-in dataset (100 emails):

| Metric    | Logistic Regression | Naive Bayes |
|-----------|--------------------:|------------:|
| Accuracy  |              ~95%   |       ~90%  |
| Precision |              ~95%   |       ~90%  |
| Recall    |              ~95%   |       ~90%  |
| F1-Score  |              ~95%   |       ~90%  |

> Results may vary slightly due to train/test split randomness.
> Add more training data in `dataset_loader.py` to improve accuracy.

---

## 📸 Screenshots

> Launch the app and take screenshots of:
> - The main web UI
> - A phishing email result (red alert)
> - A safe email result (green badge)
> - The confusion matrix charts in `reports/`

---

## 🔮 Future Enhancements

| Enhancement | Description |
|---|---|
| 📦 Larger Dataset | Train on real-world datasets like SpamAssassin or Enron |
| 🤖 Deep Learning | Use LSTM or BERT for higher accuracy |
| 📎 Attachment Scan | Detect malicious attachments in emails |
| 🌐 URL Analysis | Check URLs against known phishing blacklists |
| 📈 Active Learning | Retrain model as users flag new phishing emails |
| 🔐 Email Header Analysis | Parse DKIM/SPF/DMARC headers for spoofing detection |
| 📱 Email Client Plugin | Browser extension or Outlook add-in integration |
| 🗄️ Database Logging | Store all predictions for audit and analysis |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Scikit-learn | ML models and TF-IDF |
| NLTK | NLP preprocessing |
| Pandas / NumPy | Data handling |
| Flask | Web framework |
| Matplotlib / Seaborn | Evaluation charts |
| Joblib | Model serialization |

---

## 📄 License

This project is for educational purposes. Free to use and modify.

---

*Built with ❤️ using Python, Scikit-learn, and Flask*
