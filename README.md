# Projeto de Previsão de Atrasos de Voos - Case Machine Learning Engineer


O projeto implementa um sistema completo de previsão de atrasos de voos, demonstrando competências em  desenvolvimento de API, gerenciamento de experimentos, orquestração de tarefas e análise exploratória de dados.

## Componentes Principais

### API

A API, construída com FastAPI, oferece endpoints para:

- Previsão de atrasos de voos
- Carregamento de novos modelos
- Obtenção de métricas do modelo atual

### MLflow

O MLflow é utilizado para o gerenciamento de experimentos de Machine Learning, rastreando parâmetros, métricas e artefatos dos modelos treinados.

### Airflow

O Apache Airflow orquestra o processo de treinamento e atualização do modelo. A DAG principal (`airport_model_training_dag.py`) é responsável por:

1. Treinar modelos
2. Registrar métricas no MLflow
3. Atualizar o modelo na API

### Análise Exploratória de Dados (EDA)

O notebook Jupyter `notebooks/Case-MLOps.ipynb` contém a análise exploratória dos dados, incluindo:

- Visualizações dos dados de voos
- Análise de correlações entre variáveis
- Insights sobre fatores que influenciam atrasos

## Arquitetura do Projeto

O diagrama da arquitetura do projeto, disponível em `docs/case-diagram.png`, ilustra a interação entre os diferentes componentes do sistema, incluindo o fluxo de dados e a integração entre componentes da pipeline MLOps em ambiente AWS.

## Tecnologias Utilizadas

- FastAPI
- MLflow
- Apache Airflow
- Scikit-learn
- Pandas
- Jupyter Notebook
- Docker e Docker Compose

Este projeto demonstra a implementação de um pipeline completo de MLOps, desde a análise exploratória de dados até a implantação e monitoramento contínuo de modelos de machine learning.