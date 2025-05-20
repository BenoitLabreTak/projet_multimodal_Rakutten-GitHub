#!/bin/bash
# il faut passer le numéro de version à stocker pour les modèles en paramètre du script

version_model="${1:-0.0}"
zenml stack set local_gitflow_stack
python3 pipelines/pipeline_global.py "$version_model"

# orchestrateur docker
# zenml orchestrator register local_docker_orchestrator --flavor=local_docker
