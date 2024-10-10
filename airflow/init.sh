#!/bin/bash
set -e

# Função para aguardar o PostgreSQL ficar disponível
wait_for_postgres() {
    echo "Aguardando o PostgreSQL iniciar..."
    while ! nc -z postgres 5432; do
      sleep 1
    done
    echo "PostgreSQL está pronto!"
}

# Inicializa o banco de dados do Airflow
initialize_airflow_db() {
    echo "Inicializando o banco de dados do Airflow..."
    airflow db init
    echo "Banco de dados do Airflow inicializado com sucesso!"
}

# Cria um usuário admin
create_admin_user() {
    echo "Criando usuário admin..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
    echo "Usuário admin criado com sucesso!"
}

# Cria conexões padrão
create_default_connections() {
    echo "Criando conexões padrão..."
    # Exemplo de conexão com PostgreSQL
    airflow connections add 'postgres_default' \
        --conn-type 'postgres' \
        --conn-login 'airflow' \
        --conn-password 'airflow' \
        --conn-host 'postgres' \
        --conn-port 5432 \
        --conn-schema 'airflow'
    
    # Adicione mais conexões conforme necessário
    
    echo "Conexões padrão criadas com sucesso!"
}

# Cria variáveis padrão
create_default_variables() {
    echo "Criando variáveis padrão..."
    airflow variables set 'airports-database' '/data/airports-database.csv'
    # Adicione mais variáveis conforme necessário
    echo "Variáveis padrão criadas com sucesso!"
}

# Função principal
main() {
    wait_for_postgres
    initialize_airflow_db
    create_admin_user
    create_default_connections
    create_default_variables
    
    echo "Inicialização do Airflow concluída com sucesso!"
}

# Executa a função principal
main

# Executa o comando fornecido (se houver)
exec "$@"