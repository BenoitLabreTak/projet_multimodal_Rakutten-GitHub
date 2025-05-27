# Utilise une image officielle ZenML avec le client préinstallé
FROM zenmldocker/zenml:0.82.1-py3.12
# Définit le répertoire de travail
WORKDIR /app
# pour des raisons de cache de construction, on copie d'abord le requirements.txt
# et on installe les dépendances avant de copier le reste du code
COPY ./requirements.txt ./
RUN apt update && \
    apt install -y python3-opencv && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt && \
    zenml integration install -y s3 pytorch
# Copie le dossier pipelines dans l'image
COPY ./ ./
# Affecte les droits d'exécution
RUN chmod +x ./run.sh && \
    chmod +x ./run-firsttime.sh 
# Commande par défaut (à adapter selon le besoin)
#CMD ["/bin/bash"]
