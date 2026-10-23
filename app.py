
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

folder = Path(__file__).resolve().parent

app = FastAPI()

app.mount("/static", StaticFiles(directory=folder / "static"), name="static")
templates = Jinja2Templates(directory=str(folder / "templates"))

reg_saved = joblib.load(folder / "house_price_model.joblib")
class_saved = joblib.load(folder / "classification_model.joblib")

reg_model = reg_saved["model"]
class_model = class_saved["model"]

reg_columns = reg_saved["feature_columns"]
class_columns = class_saved["feature_columns"]

class HouseInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=0)
    bathrooms: int = Field(ge=0)
    stories: int = Field(ge=0)
    parking: int = Field(ge=0)
    furnishingstatus: Literal["furnished", "semi-furnished", "unfurnished"]
    mainroad: Literal["yes", "no"]
    guestroom: Literal["yes", "no"]
    basement: Literal["yes", "no"]
    hotwaterheating: Literal["yes", "no"]
    airconditioning: Literal["yes", "no"]
    prefarea: Literal["yes", "no"]

class DealInput(HouseInput):
    asking_price: float = Field(gt=0)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict/regression")
def predict_regression(data: HouseInput):
    house = pd.DataFrame([data.model_dump()], columns=reg_columns)

    price = reg_model.predict(house)[0]

    return {
        "prediction": round(float(price), 2)
    }

@app.post("/predict/classification")
def predict_classification(data: DealInput):
    house = pd.DataFrame([data.model_dump()], columns=class_columns)

    result = int(class_model.predict(house)[0])

    categories = {
        0: "Potential Bargain",
        1: "Fair Price",
        2: "Potentially Overpriced"
    }

    return {
        "prediction": categories[result]
    }
