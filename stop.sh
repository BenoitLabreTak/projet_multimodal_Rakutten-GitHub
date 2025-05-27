#!/bin/bash

echo "🛑 Arrêt des services MLOps..."

# Arrêt des conteneurs Docker
docker-compose down

echo "✅ Tous les services Docker ont été arrêtés."
