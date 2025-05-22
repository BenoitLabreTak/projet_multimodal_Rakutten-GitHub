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
  pip install pip-tools pytest zenml
  ```

### 2. Installation des dépendances

```bash
pip-compile requirements.in --output-file=requirements.txt
pip install -r requirements.txt
```

### 3. Lancement des services

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

### Exemple de pipeline :

```python
@pipeline
def auto_eval_and_retrain_pipeline():
    metrics = evaluate_model()
    retrain_triggered = conditional_retrain(metrics)
    notify_slack_on_success(retrain_triggered)
```

### Exécution

```bash
zenml init
zenml pipeline run auto_eval_and_retrain_pipeline
```

---

## 📁 Arborescence

```
rakuten_mlops/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   └── services/
├── tests/
├── models/
├── data/
├── Dockerfile
├── docker-compose.yml
├── requirements.in
├── requirements.txt
├── pytest.ini
└── README.md
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