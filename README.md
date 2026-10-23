# Full-Stack Machine Learning App

I built a FastAPI web app using Scikit-learn and the Housing dataset.

- **Regression:** Predicts house prices.
- **Classification:** Predicts Potential Bargain, Fair Price, or Potentially Overpriced using estimated values and asking prices.

Both models are saved using Joblib. R² and accuracy are calculated in `train_model.ipynb`.

## How to Run

1. `pip install -r requirements.txt`
2. `python -m uvicorn app:app --reload`
3. Open `http://127.0.0.1:8000`

**API:** `/predict/regression` and `/predict/classification`

**Dataset:** https://www.kaggle.com/datasets/yasserh/housing-prices-dataset