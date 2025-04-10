from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

# загрузка модели
model = joblib.load('laptop_price_model.pkl')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # чтение CSV-файла
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    
    # предсказание
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)