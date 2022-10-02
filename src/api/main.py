import sys
sys.path.append("..")
import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 
# Import ml stuff
from ml.model import inference_api

# Instantiate the app.
app = FastAPI()
    
# Declare the data object with its components and their type.
class CensusObject(BaseModel):
    workclass: str
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias="native-country")
    age: float
    fnlgt: float
    education_num: float = Field(alias="education-num")
    capital_gain: float = Field(alias="capital-gain")
    capital_loss: float = Field(alias="capital-loss")
    hours_per_week: float = Field(alias="hours-per-week")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                'age': 30.0, 
                'workclass': 'Private', 
                'fnlgt': 241259.0, 
                'education': 'Assoc-voc', 
                'education-num': 11.0, 
                'marital-status': 'Married-civ-spouse',
                'occupation': 'Transport-moving', 
                'relationship': 'Husband', 
                'race': 'White', 
                'sex': 'Male', 
                'capital-gain': 0.0, 
                'capital-loss': 0.0, 
                'hours-per-week': 40.0, 
                'native-country': 'United-States'
            }
        }


def census_to_df(census_object: CensusObject) -> pd.DataFrame:
    df = pd.DataFrame([census_object.dict()])
    for col in df.columns:
        col_new = col.replace('_', '-')
        df.rename(columns={col: col_new}, inplace=True)
    df = df[['age', 'workclass', 'fnlgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]
    return df


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "This server is for performing inference on census data to predict whether a person's salary exceeds 50k USD"}


# This allows sending of data (our CensusObject) via POST to the API.
@app.post("/infer")
async def create_item(census_object: CensusObject):
    df = census_to_df(census_object)
    res = inference_api(df)
    return {'result': bool(res)}
