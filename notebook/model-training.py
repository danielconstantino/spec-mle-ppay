import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pickle

# Ler o CSV
df = pd.read_csv("../notebook/airports-database.csv")

# Tratar valores nulos
df = df.dropna()

# Criar coluna target (1 se houve atraso, 0 caso contrário)
df['delayed'] = (df['arr_delay'] > 0).astype(int)

# Selecionar features relevantes
features = ["month", "day", "dep_time", "distance", "carrier"]

# Preparar os dados
X = df[features]
y = df['delayed']

# Dividir dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar pipeline para pré-processamento
numeric_features = ["month", "day", "dep_time", "distance"]
categorical_features = ["carrier"]

numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

# Criar e configurar o classificador
rf = RandomForestClassifier(n_estimators=10, random_state=42)

# Montar pipeline completa
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

# Treinar o modelo
model = pipeline.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Avaliar o modelo
auc = roc_auc_score(y_test, probabilities)
print(f"AUC: {auc}")

# Calcular acurácia
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Salvar o modelo treinado
model_path = "../models/current_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Modelo salvo em: {model_path}")