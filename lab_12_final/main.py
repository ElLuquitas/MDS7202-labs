from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import xgboost as xgb
import uvicorn

# Cargar el modelo desde el archivo .pkl
model_path = "models/final_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Crear la aplicación FastAPI
app = FastAPI()

# Modelo de entrada para validación con Pydantic
class InputData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Ruta de inicio
@app.get("/")
async def home():
    return {
        "modelo": "XGBoost con DMatrix",
        "problema": "Determinar si el agua es potable",
        "entrada": {
            "ph": "float",
            "Hardness": "float",
            "Solids": "float",
            "Chloramines": "float",
            "Sulfate": "float",
            "Conductivity": "float",
            "Organic_carbon": "float",
            "Trihalomethanes": "float",
            "Turbidity": "float",
        },
        "salida": {"potabilidad": "0 (no potable) o 1 (potable)"}
    }

@app.post("/potabilidad/")
async def predict_potabilidad(data: InputData):
    # Convertir los datos de entrada a formato adecuado para el modelo
    input_data = [[
        data.ph, data.Hardness, data.Solids, data.Chloramines,
        data.Sulfate, data.Conductivity, data.Organic_carbon,
        data.Trihalomethanes, data.Turbidity
    ]]

    # Nombres de las columnas esperadas
    column_names = [
        "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
    ]

    # Convertir a DMatrix con las columnas adecuadas
    dmatrix_data = xgb.DMatrix(input_data, feature_names=column_names)

    # Realizar la predicción con el modelo cargado
    prediction = model.predict(dmatrix_data)

    # Convertir la predicción a una respuesta JSON
    return {"potabilidad": int(round(prediction[0]))}


# Ejecutar el servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)

