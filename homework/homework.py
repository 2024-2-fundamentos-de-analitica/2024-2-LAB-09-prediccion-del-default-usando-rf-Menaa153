# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#



import pandas as pd
import numpy as np
import os
import json
import gzip
import joblib
import zipfile

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def preprocess_data(zip_file_path):
    """ Lee y limpia los datos directamente desde un archivo ZIP. """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extrae el nombre del archivo CSV dentro del ZIP
        csv_filename = zip_ref.namelist()[0]
        with zip_ref.open(csv_filename) as f:
            df = pd.read_csv(f)

    # Renombrar columna objetivo y eliminar columna ID
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)

    # Eliminar valores faltantes
    df.dropna(inplace=True)

    # Agrupar EDUCATION > 4 en la categoría "others"
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df

def split_data(df):
    """ Divide los datos en X (características) e y (etiquetas) """
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y

from sklearn.compose import ColumnTransformer

def build_pipeline():
    """ Crea un pipeline con OneHotEncoding y RandomForestClassifier """
    
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    
    preprocessor = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    
    return pipeline


def optimize_hyperparameters(pipeline, X_train, y_train):
    """ Optimización de hiperparámetros usando GridSearchCV con validación cruzada. """
    param_grid = {
        "classifier__n_estimators": [100, 300],  # Aumenta el número de árboles
        "classifier__max_depth": [10, 20, None],  # Prueba sin límite de profundidad
        "classifier__min_samples_split": [2, 5],  # Reducir sobreajuste
        "classifier__min_samples_leaf": [1, 2],  # Variar tamaño mínimo de hojas
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Mejor precisión encontrada: {grid_search.best_score_}")
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")



    return grid_search  # Retorna el objeto GridSearchCV completo



def save_model(model, file_path):
    """ Guarda el modelo optimizado (GridSearchCV) en gzip """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"🔍 Guardando modelo de tipo: {type(model)}")
    with gzip.open(file_path, "wb") as f:
        joblib.dump(model, f)  # Guardamos el objeto GridSearchCV completo


def calculate_metrics(model, X, y, dataset_type):
    """ Calcula precisión, recall, f1-score y matriz de confusión """
    y_pred = model.predict(X)

    metrics = {
        "type": "metrics",  # 🔹 Agregamos este campo obligatorio
        "dataset": dataset_type,
        "precision": precision_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }

    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type": "cm_matrix",  # 🔹 Agregamos este campo obligatorio
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

    return metrics, cm_dict

def save_metrics(metrics_list, file_path):
    """ Guarda las métricas en un archivo JSON con el orden correcto """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    ordered_metrics = []  # Lista ordenada de métricas

    # Añadir primero las métricas
    for metric in metrics_list:
        if metric["type"] == "metrics":
            ordered_metrics.append(metric)
    
    # Luego añadir las matrices de confusión
    for metric in metrics_list:
        if metric["type"] == "cm_matrix":
            ordered_metrics.append(metric)

    # Guardar en JSON
    with open(file_path, "w", encoding="utf-8") as f:
        for metric in ordered_metrics:
            f.write(json.dumps(metric) + "\n")



def main():
    # Paso 1: Cargar y limpiar datos
    train_file = "../files/input/train_data.csv.zip"
    test_file = "../files/input/test_data.csv.zip"

    train_df = preprocess_data(train_file)
    test_df = preprocess_data(test_file)

    # Paso 2: Dividir datos en X e y
    X_train, y_train = split_data(train_df)
    X_test, y_test = split_data(test_df)

    # Paso 3: Construir pipeline
    pipeline = build_pipeline()

    # Paso 4: Optimizar hiperparámetros
    model = optimize_hyperparameters(pipeline, X_train, y_train)  # Retorna GridSearchCV

    # Paso 5: Guardar modelo
    model_path = "../files/models/model.pkl.gz"
    save_model(model, model_path)

    # Paso 6 y 7: Calcular métricas y matriz de confusión
    metrics_train, cm_train = calculate_metrics(model.best_estimator_, X_train, y_train, "train")
    metrics_test, cm_test = calculate_metrics(model.best_estimator_, X_test, y_test, "test")

    # Guardar métricas en JSON
    metrics_path = "../files/output/metrics.json"
    save_metrics([metrics_train, cm_train, metrics_test, cm_test], metrics_path)

    print(f"Modelo guardado en {model_path}. Métricas en {metrics_path}.")
    print(f"Precisión en train: {model.score(X_train, y_train)}")
    print(f"Precisión en test: {model.score(X_test, y_test)}")


if __name__ == "__main__":
    main()