import sys
import os
from uvicorn import run

# Ajouter le dossier courant (racine du projet) au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("L'API ne supporte qu'un worker avec ce mode de lancement")
    run("app.main:app", host="0.0.0.0", port=8000, reload=True)