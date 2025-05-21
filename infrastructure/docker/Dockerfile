# Dockerfile de base
FROM python:3.12-slim
WORKDIR /project
# pour des raisons de cache de construction, on copie d'abord le requirements.txt
# et on installe les dépendances avant de copier le reste du code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# copie du code de l'application. Elle est configurée pour chercher les données dans /data
COPY ./run.py .
COPY ./app ./app

CMD ["python", "run.py"]
