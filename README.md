# 🧠 Rakuten MLOps Project

## 🔍 Objectif

Ce projet met en œuvre un pipeline MLOps complet pour la classification de produits à partir de texte et d'images, incluant :

- APIs FastAPI pour le prétraitement, la prédiction et le réentraînement
- Monitoring via Prometheus & Grafana
- Tests unitaires automatisés avec Pytest
- Pipeline ZenML : évaluation automatique, retrain conditionnel et alertes Slack

---

## 🚀 Lancement local

### 1. Prérequis

- Docker & Docker Compose
- Python 3.12+
- Outils Python :
  ```bash
  pip install pip-tools pytest zenml[server] dvc dvc-s3
  ```
  ou
  ```bash
  pip install -r requirements-dev.txt
  ```
- Récupération des données depuis DVC (voir les commandes d'authentification DVC sur [https://dagshub.com/BenoitLabreTak/projet_multimodal_Rakutten-GitHub](https://dagshub.com/BenoitLabreTak/projet_multimodal_Rakutten-GitHub)), puis lancer la commande:
```bash
dvc pull
```

### 2. Installation des dépendances

```bash
pip-compile requirements.in --output-file=requirements.txt
pip install -r requirements.txt
```

### 3. Configurer l'envoi de message sur Slack

#### Créer une url webhook Slack
- Aller sur [https://api.slack.com/apps/](https://api.slack.com/apps/)
- identifiez-vous
- Créer une nouvelle application vide (new app from scratch). 
- Dans le menu "Settings" sur la gauche, cliquez sur "Incoming Webhooks"
- Activez "Activate Incoming Webhooks" et récupérez l'url indiquée

#### Configurez Alertmanager
- Recopier le fichier `monitoring/alertmanager/config-example.yml` vers `monitoring/alertmanager/config.yml` (Le fichier config.yml n'est pas suivi par Git car il contient des infos sensibles)
- compléter les informations nécessaires dans le fichier `config.yml`: 
  - `username`: le nom d'utilisateur Slack qui a été utilisé lors de l'étape précédente
  - `api_url`: l'url récupérée lors de l'étape précédente

#### Configurez le pipeline ZenML
- définissez une variable d'environnement nommée `SLACK_WEBHOOK_URL` contenant l'url récupérée.

### 4. Lancement des services

```bash
docker-compose up --build
```

- **API Swagger** : [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus** : [http://localhost:9090](http://localhost:9090)
- **Grafana** : [http://localhost:3000](http://localhost:3000)
  - Login : `admin` / Password : `admin` (par défaut)

---

## 📡 Endpoints principaux

| Méthode | Route                                 | Description                            |
|--------:|:--------------------------------------|:----------------------------------------|
| POST    | `/predict/text/manual`                | Prédiction CamemBERT sur désignation + description |
| POST    | `/predict/image/manual`               | Prédiction ResNet à partir d’une image  |
| POST    | `/train/text`                         | Réentraînement du modèle texte          |
| POST    | `/train/image`                        | Réentraînement du modèle image          |
| GET     | `/test/preprocess/text/manual`        | Lance le test unitaire associé          |
| GET     | `/metrics`                            | Exposition Prometheus                   |

---

## 🧪 Tests unitaires

```bash
pytest tests/
```
ou
```bash
docker compose run api pytest tests/
```

Lancement via API (Swagger ou cURL) :

```http
GET /test/preprocess/text/manual
```

Sortie JSON :
```json
{
  "exit_code": 0,
  "stdout": "...",
  "stderr": "",
  "success": true
}
```

---

## 📈 Monitoring

- Toutes les routes sont instrumentées avec `prometheus_fastapi_instrumentator`.
- **Prometheus** collecte les métriques.
- **Grafana** peut être configuré avec un dashboard personnalisé (ex : latence, taux de succès, requêtes/s).
  exemple de dashboard à importer: https://grafana.com/grafana/dashboards/18739-fastapi-observability/
  exemple de dashboard à importer pour les métriques conteneurs: https://grafana.com/grafana/dashboards/10619-docker-host-container-overview/ 

---

## 🔁 Pipeline ZenML

### Fonctionnalités

- Évaluation automatique (1% du dataset)
- Réentraînement déclenché uniquement si le score F1 est insuffisant
- Notification Slack si un nouveau modèle est sauvegardé

### Exécution
- s'assurer d'avoir démarrer le docker-compose avant

```bash
zenml init
# Pour envoyer des messages Slack, modifiez ci-dessous l'url "Incoming webhook" (voir étape 3)
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxxx/yyyyy
python pipelines/text_auto_eval_and_retrain_pipeline.py
python pipelines/image_auto_eval_and_retrain_pipeline.py
# consulter le dashboard:
zenml login --local --port 9000
```

### Refactorisation
POC Réentrainement des modèles sans utilisation de l'API.
Avantages:
- n'impacte pas la disponibilité de l'API aux utilisateurs finaux
- n'expose pas sur Internet les processus de réentrainement
- utilise les artefacts pour conserver les étapes intermédiaires:
  - permet de faire du drift monitoring en comparant plusieurs versions de modèles et données
  - permet de reprendre des réentrainements interrompus
- peut être paramétré pour exécuter les steps avec Docker, Airflow, Kubernetes pour plus de scalabilité, de paramétrage de ressources cpu, et d'efficience
Inconvénients:
- Il faut réimplémenter le processus de réentrainement qui l'est déjà dans l'API
Voir [pipeline_refactor/README.md](pipeline_refactor/README.md)

---

## 📁 Arborescence

```
rakuten_mlops/
├── app/
│   ├── api/
│   ├── core/ : fichier de configuration de l'API
│   ├── models/ : les modèles utilisés ou générés par l'application
│   └── services/
├── tests/
├── data/
├── monitoring/ : fichiers de paramétrages liés au monitoring
├── pipelines/ : fichiers liés au pipeline de réentraineement
├── pipelines_refactor/ : fichiers liés au pipeline de réentraineement détaillé
├── Dockerfile
├── docker-compose.yml
├── requirements.in
├── requirements.txt
├── requirements-dev.txt : modules nécessaires pour exécuter le pipeline de réentrainement sur le poste
├── pytest.ini
├── README.md
└── run.py : lancement de l'application app
```

---

## ✅ À faire

- [ ] Ajouter dashboards Grafana par défaut
- [ ] Intégration continue (GitHub Actions)
- [ ] Tracking modèle complet via DagsHub ou MLflow
- [ ] Trigger automatique des tests en push/pull request

---

## 🤖 Auteur

**Mehdi Malhas**  
> Machine Learning Engineer | MLOps  
> [LinkedIn](https://www.linkedin.com/in/mehdi-malhas)

**Fabrice Moreau**
> Machine Learning Engineer | MLOps  
> [LinkedIn](https://www.linkedin.com/in/fabrice-moreau)

**Benoit Labre-Takam**

**Nicolas Haddad**
