#!/bin/bash

echo "🚀 Démarrage du projet MLOps avec CRON + Docker + pipelines locaux..."

# 1. Chemin absolu sécurisé
PROJECT_DIR="$(pwd)"
PYTHON_EXEC="$(which python3)"

# 2. Chemins vers les pipelines (avec protection)
TEXT_PIPELINE="${PROJECT_DIR}/pipelines/text_auto_eval_and_retrain_pipeline.py"
IMAGE_PIPELINE="${PROJECT_DIR}/pipelines/image_auto_eval_and_retrain_pipeline.py"
LOG_DIR="${PROJECT_DIR}/logs"

# 3. Créer le dossier de logs
mkdir -p "$LOG_DIR"

# 4. Préparer les entrées cron avec `env` pour éviter les erreurs d'espaces
CRON_ENTRY_TEXT="0 */4 * * * env PYTHONPATH='$PROJECT_DIR' \"$PYTHON_EXEC\" \"$TEXT_PIPELINE\" >> \"$LOG_DIR/cron_text.log\" 2>&1"
CRON_ENTRY_IMAGE="30 */4 * * * env PYTHONPATH='$PROJECT_DIR' \"$PYTHON_EXEC\" \"$IMAGE_PIPELINE\" >> \"$LOG_DIR/cron_image.log\" 2>&1"

# 5. Vérifier et ajouter si manquant
CURRENT_CRON=$(crontab -l 2>/dev/null || echo "")

if ! echo "$CURRENT_CRON" | grep -Fq "$TEXT_PIPELINE"; then
    (echo "$CURRENT_CRON"; echo "$CRON_ENTRY_TEXT") | crontab -
    echo "✅ Cron texte ajouté"
else
    echo "ℹ️ Cron texte déjà présent"
fi

if ! echo "$CURRENT_CRON" | grep -Fq "$IMAGE_PIPELINE"; then
    (crontab -l; echo "$CRON_ENTRY_IMAGE") | crontab -
    echo "✅ Cron image ajouté"
else
    echo "ℹ️ Cron image déjà présent"
fi

# 6. Démarrer Docker Compose
echo "🐳 Lancement des conteneurs Docker..."
docker-compose up --build -d


echo "✅ Tous les services sont opérationnels. Pipelines et cron actifs."