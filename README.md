# Full-Stack Machine Learning App

Salam Sir, I made a simple web app with two machine learning models using Scikit-learn and FastAPI. The regression model predicts house prices using the Kaggle Housing dataset, and the classification model predicts Iris flower species. Both models are saved with Joblib.

On the test data, the house price model achieved **R² = 0.653**, and the Iris model achieved **accuracy = 96.7%** (random_state=42).

## How to run

1. Open a terminal in this folder.
2. Install libraries: `pip install -r requirements.txt`
3. (Optional) Train both models again: `python train_models.py`
4. Start the server: `python -m uvicorn app:app --reload`
5. Open **http://127.0.0.1:8000** to use both forms.

The API has two JSON POST endpoints: `/predict/regression` and `/predict/classification`. You can also test them at **http://127.0.0.1:8000/docs**.

**Housing dataset:** https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

**Note:** House price predictions use the original dataset's currency, not automatically PKR. The notebooks from my earlier regression work are also included.
