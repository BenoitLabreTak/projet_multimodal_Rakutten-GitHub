# Pour démarrer un process de réentrainement

# Architecture
Le docker-compose va lancer un serveur dédié ZenML et ses dépendances (serveur s3 minio pour stockage artefacts et serveur mysql pour stockage des données de fonctionnement de zenml)
On relance un conteneur avec le pipeline à chaque fois qu'on veut l'exécuter

# Prérequis
Docker (docker compose v2)

# Configuration
Recopier le fichier `example.env` en `.env`.
Vous pouvez adapter son contenu pour personnaliser les logins/mots de passe à utiliser par le serveur zenml lancé.
# lancement serveur 
```bash
docker compose up -d
```

# Accès
Vous devriez avoir accès à l'interface web de ZenML sur [http://localhost:8080](http://localhost:8080) au bout de quelques instants.
Il est possible de se connecter à Minio sur [http://localhost:9001](http://localhost:9001)

# Etapes nécessaires avant lancement d'un pipeline
Ces étapes sont executées depuis le conteneur pipeline. Vous pouvez les exécuter directement sur votre poste, mais il faudra alors exécuter les commandes présentes dans `docker/Dockerfile`.

Ces étapes effectue la configuration serveur, **il ne faut les exécuter qu'une seule fois**, les réglages sont conservés dans que les volumes Docker du docker-compose sont conservés sur la machine.

## Création du bucket `zenml-bucket`
Il faut vous connecter à Minio sur [http://localhost:9001](http://localhost:9001) en renseignant le login/mot de passe (présent dans votre fichier `.env` : `MINIO_ROOT_USER` et `MINIO_ROOT_PASSWORD`). 
Aller dans le menu `Buckets` et cliquez sur le bouton à droite `Create Bucket +`.
Vous devez alors renseigner comme `Bucket Name` la valeur `zenml-bucket`. Vous pouvez laisser les autres valeurs telles quelles et cliquer sur le bouton `Create Bucket`.


## Authentification sur le serveur ZenML
# Via variable d'environnement
Les variables d'environnements définies dans le conteneur pipeline du docker-compose `ZENML_STORE_URL`, `ZENML_STORE_USERNAME`, `ZENML_STORE_PASSWORD` permettent de ne pas avoir à s'authentifier.

## Via la commande login
Pour vous authentifier depuis votre autre poste sans définir de variable d'environnement, vous lancer la commande
```bash
zenml login http://localhost:8080
``
Elle affiche le message suivant:
```console
Authenticating to ZenML server 'http://zenml-server:8080' using the web login...
If your browser did not open automatically, please open the following URL into your browser to proceed with the authentication:

http://zenml-server:8080/devices/verify?device_id=fune-chaine-aleatoire
```

Il faut récupérer l'url indiquée complète, remplacer la valeur `zenml-server` par `localhost` (zenml-server est un nom qui n'est résolu que pour l'intérieur du réseau Docker, pas sur notre machine) et ouvrir un navigateur.
Renseigner alors le login et mot de passe tels qu'indiqués dans le fichier `.env` dans les variables `ZENML_USERNAME` et `ZENML_PASSWORD`.
Cette étape sera nécessaire à chaque conteneur lancé qui a besoin de se connecter au serveur ZenML. Il serait possible de créer une clé d'API pour faciliter.

## Création de la stack et artifact-store dans ZenML

Lancer la commande suivante:
```bash
docker compose run --rm pipeline /app/run-firsttime.sh 
```

# Récupérer les données avec Dagshub
A compléter:
Le script a besoin de récupérer les données dans les dossiers parents `../models`, `../images` et `../datasets`.

# Lancer un pipeline
Vous pouvez remplacer `version_modele` par n'importe quelle chaîne de caractère qui représente le nom de version qui sera associé aux modèles par ZenML.
```bash
docker compose run --rm pipeline /app/run.sh version_modele
```

# Modifier le pipeline
A chaque modification du pipeline, il est nécessaire d'actualiser l'image Docker construite qui le contient si vous souhaitez déployer ce conteneur.
Cela peut être fait avec la commande
```bash
docker compose build
```

Il est possible de travailler dans le conteneur avec la commande
```bash
docker compose run --rm -it pipeline bash
```
