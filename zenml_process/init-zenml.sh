#!/bin/bash
# Script d'initialisation pour zenml-server (création du service account)

# Récupération du paramètre MODEL_VERSION, valeur par défaut "0.0"
toto="${1:-0.0}"

# Exécute la commande ZenML pour créer le service account
zenml service-account create myserviceaccount --output-file /zenml-key/myserviceaccount

# Lancement du serveur ZenML (commande par défaut)
exec zenml server start
