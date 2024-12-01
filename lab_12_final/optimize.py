from sklearn.model_selection import train_test_split
import os
import json
import pickle
from datetime import datetime
import optuna
import mlflow
import xgboost as xgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow # importar mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


# Función para optimizar el modelo
def optimize_model(X_train, X_valid, y_train, y_valid):
    """Optimiza hiperparámetros de XGBoost y registra resultados en MLflow."""

    # Configuración de experimentos
    experiment_name = "XGBoost_Optimization_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_mlflow_experiment(experiment_name)

    # Función objetivo para Optuna
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "max_leaves": trial.suggest_int("max_leaves", 0, 100),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        }

        # Iniciar un "run" en MLflow
        try:
            with mlflow.start_run(run_name=f"XGBoost_lr_{params['learning_rate']}_depth_{params['max_depth']}"):
                # Preparar datos
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dvalid = xgb.DMatrix(X_valid, label=y_valid)

                # Entrenar el modelo
                model = xgb.train(
                    params, dtrain, num_boost_round=params["n_estimators"],
                    evals=[(dvalid, "validation")],
                    early_stopping_rounds=10, verbose_eval=False
                )

                # Predicciones y cálculo del F1-score
                preds = model.predict(dvalid)
                preds_binary = (preds > 0.5).astype(int)
                f1 = f1_score(y_valid, preds_binary)

                # Registrar métricas y parámetros
                mlflow.log_metric("valid_f1", f1)
                mlflow.log_params(params)

                # Guardar el modelo en MLflow
                mlflow.sklearn.log_model(model, artifact_path="model")

                return f1
        finally:
            mlflow.end_run()  # Asegurar cierre del run

    # Crear y optimizar el estudio
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Registrar gráficos en MLflow
    with mlflow.start_run(run_name="Optuna Analysis"):
        plot_dir = "./plots"
        history_path, importance_path = save_optuna_plots(study, plot_dir)
        mlflow.log_artifact(history_path, artifact_path="plots")
        mlflow.log_artifact(importance_path, artifact_path="plots")

    # Obtener el mejor modelo usando la función `get_best_model`
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    best_model = get_best_model(experiment_id)

    # Guardar el modelo serializado y configuraciones
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact(model_path, artifact_path="models")

    config_path = os.path.join(model_dir, "best_params.json")
    with open(config_path, "w") as f:
        json.dump(study.best_params, f)
    mlflow.log_artifact(config_path, artifact_path="configs")

    # Guardar gráfico de importancia de las variables
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(best_model)
    feature_importance_path = os.path.join(plot_dir, "feature_importance.png")
    plt.savefig(feature_importance_path)
    mlflow.log_artifact(feature_importance_path, artifact_path="plots")
    plt.close()

    print("Optimización completada. Mejor f1-score:", study.best_value)
    return best_model

if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('water_potability.csv')
    # Crear una copia para preservar el DataFrame original (opcional)
    data_scaled = data.copy()

    # Seleccionar las columnas a escalar
    columns_to_scale = [col for col in data.columns if col != 'Potability']

    # Crear y ajustar el escalador
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    X = data_scaled.drop(columns=['Potability'])
    y = data['Potability']

    # Dividir en conjunto de entrenamiento y validación
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Llamar a la función de optimización
    optimize_model(X_train, X_valid, y_train, y_valid)