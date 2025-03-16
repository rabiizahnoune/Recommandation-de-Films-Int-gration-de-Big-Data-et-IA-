# Système de Recommandation de Films

## 1. Aperçu du Projet

Le **Système de Recommandation de Films** est une application basée sur les données MovieLens (100k) qui génère des recommandations de films personnalisées. Il utilise **Apache Spark** pour le traitement des données, **Elasticsearch** pour le stockage et la recherche, et une **API Flask** pour l'interaction utilisateur. Le système est déployé via **Docker** pour une architecture scalable et inclut une interface utilisateur web.

### Objectifs
- Fournir des recommandations personnalisées pour les utilisateurs existants (basées sur leurs historiques) et nouveaux (via prédictions démographiques).
- Assurer la scalabilité avec Spark et des recherches rapides via Elasticsearch.
- Offrir une API REST et une interface utilisateur simple.

### Technologies
- **Traitement** : Apache Spark (PySpark)
- **Stockage/Recherche** : Elasticsearch 8.5.1
- **Visualisation** : Kibana 8.5.1
- **Backend** : Flask (Python)
- **Modélisation** : Scikit-learn (RandomForestRegressor)
- **Conteneurisation** : Docker, Docker Compose
- **Frontend** : HTML/CSS

---

## 2. Architecture du Projet

### Structure des Répertoires
- **Racine** : `C:\Users\Youcode\Documents\Recommandation de Films`
  - `.gitignore` : Exclusion de `myenv` et `data/raw/ml-100k`.
  - `commande.sh` : Script pour exécuter le prétraitement Spark.
  - `docker-compose.yaml` : Configuration des services Docker.
  - `README.md` : Notes sur l'intégration Spark-Elasticsearch.
  - `test.py` : Génération de la structure du projet.
- **/backend** : API Flask et interface utilisateur.
  - `.env` : Variables d’environnement (ex. `ELASTICSEARCH_URL`).
  - `app.py` : API Flask avec endpoints.
  - `Dockerfile` : Configuration du conteneur Flask.
- **/spark_jobs** : Scripts Spark.
  - `/systeme/genre_predictor.pkl` : Modèle entraîné.
  - `/systeme/train_genre_model.py` : Entraînement du modèle.
- **/data/raw/ml-100k** : Données MovieLens (ex. `u.data`).

### Architecture Technique
1. **Spark Cluster** :
   - **Master** : Ports 7077, 8080, 8888.
   - **Workers** : 2 replicas, 0.5 CPU, 2 Go RAM.
2. **Elasticsearch** : Stockage des profils et films (port 9200).
3. **Kibana** : Visualisation (port 5601).
4. **Backend Flask** : API et UI (port 5000).
5. **Réseau** : `spark-es-network` (overlay).

---

## 3. Fonctionnalités Principales

### 3.1. Prétraitement des Données
- **Script** : `train_genre_model.py`
- **Entrée** : MovieLens (`u.data`, `u.item`, `u.user`).
- **Processus** :
  1. Chargement avec PySpark.
  2. Calcul des moyennes par genre par utilisateur.
  3. Stockage dans Elasticsearch (`user_profiles`, `movies`).
  4. Entraînement d’un modèle RandomForestRegressor.
- **Sortie** : Modèle (`genre_predictor.pkl`) prédisant les préférences par genre.

### 3.2. API de Recommandation
- **Fichier** : `app.py`
- **Endpoints** :
  - `/` : Interface utilisateur HTML.
  - `/recommend_session` : Recommandations pour un utilisateur.
    - **Utilisateur existant** : Utilise les profils dans `user_profiles`.
    - **Nouvel utilisateur** : Prédit avec le modèle.
  - `/results` : Affiche les résultats.
- **Méthode** : Requêtes `function_score` dans Elasticsearch.

### 3.3. Interface Utilisateur
- **Pages** :
  - `/` : Formulaires pour utilisateurs existants/nouveaux.
  - `/results` : Liste des recommandations.

---

## 4. Déploiement

### Prérequis
- Docker et Docker Compose.
- Données MovieLens dans `/data/raw/ml-100k`.

### Étapes
1. **Construire les images** :
   ```bash
   docker-compose build
