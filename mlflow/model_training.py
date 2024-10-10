import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import joblib
import requests
import json
import time

def train_model():  
    # Configuração do MLflow
    mlflow.set_tracking_uri("http://mlflow:5555")
    mlflow.set_experiment("Flight Delay Prediction - Rand. Forest")

    # Carregar apenas uma amostra dos dados
    SAMPLE_SIZE = 10000

    data = pd.read_csv("/app/notebook/airports-database.csv", nrows=SAMPLE_SIZE)

    # Converter colunas de ponto flutuante
    float_columns = ['dep_time', 'arr_time', 'arr_delay', 'air_time']
    for col in float_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Definir features numéricas e categóricas
    numeric_features = ['distance', 'dep_time', 'air_time']
    categorical_features = ['carrier', 'origin', 'dest', 'month', 'day_of_week']

    # Adicionar day_of_week
    data['day_of_week'] = pd.to_datetime(data[['year', 'month', 'day']]).dt.dayofweek

    # Definir o target
    target = 'arr_delay'

    # Remover linhas com NaN no target
    data_clean = data.dropna(subset=[target] + numeric_features + categorical_features)

    # Preparar X e y
    X = data_clean[numeric_features + categorical_features]
    y = data_clean[target]

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar um preprocessador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Criar um pipeline com o preprocessador e o modelo Random Forest
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42))
    ])

    # Função para calcular a "accuracy" de regressão
    def regression_accuracy(y_true, y_pred, tolerance=15):
        return np.mean(np.abs(y_true - y_pred) <= tolerance) * 100

    # Função para calcular a "precision" de regressão
    def regression_precision(y_true, y_pred):
        return 1 / (1 + np.mean(np.abs((y_true - y_pred) / y_true))) * 100

    # Treinar o modelo
    with mlflow.start_run():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Fazer previsões e calcular métricas
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)  # tempo médio por previsão
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)
        
        # Calcular MAPE
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        # Calcular "accuracy" e "precision" de regressão
        reg_accuracy = regression_accuracy(y_test, predictions)
        reg_precision = regression_precision(y_test, predictions)
        
        # Feature importance
        feature_importance = model.named_steps['regressor'].feature_importances_
        
        # Obter nomes das features após o one-hot encoding
        ohe = model.named_steps['preprocessor'].named_transformers_['cat']
        if hasattr(ohe, 'get_feature_names_out'):
            cat_feature_names = ohe.get_feature_names_out(categorical_features)
        else:
            cat_feature_names = ohe.get_feature_names(categorical_features)
        
        feature_names = numeric_features + cat_feature_names.tolist()
        feature_importance_dict = dict(zip(feature_names, feature_importance))
        
        # Logar parâmetros e métricas
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 20)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("explained_variance", evs)
        mlflow.log_metric("regression_accuracy", reg_accuracy)
        mlflow.log_metric("regression_precision", reg_precision)
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("avg_inference_time", inference_time)
        
        # Salvar o modelo
        model_path = "/app/models/current_model.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Salvar as métricas em um arquivo JSON
        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "explained_variance": evs,
            "regression_accuracy": reg_accuracy,
            "regression_precision": reg_precision,
            "training_time": training_time,
            "avg_inference_time": inference_time,
            "n_estimators": 100,
            "max_depth": 20
        }
        metrics_path = "/app/models/model_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")
        
        # Enviar o modelo para a API
        files = {'file': open(model_path, 'rb')}
        response = requests.post("http://api:8000/model/load/", files=files)
        if response.status_code == 200:
            print("Model successfully loaded into the API")
        else:
            print(f"Failed to load model into API. Status code: {response.status_code}")

    print("Training completed.")

if __name__ == "__main__":
    train_model()