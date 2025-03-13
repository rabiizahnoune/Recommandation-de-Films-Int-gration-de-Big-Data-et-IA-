# from flask import Flask, request, jsonify
# from elasticsearch import Elasticsearch
# from dotenv import load_dotenv
# import os
# import logging

# # Configurer les logs pour débogage
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Charger les variables d'environnement
# load_dotenv()

# # Initialiser Flask
# app = Flask(__name__)

# # Connexion à Elasticsearch
# es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
# es = Elasticsearch([es_url])

# # Vérifier la connexion à Elasticsearch
# if not es.ping():
#     raise ValueError("Impossible de se connecter à Elasticsearch !")
# else:
#     logger.info("Connexion à Elasticsearch réussie.")

# # Route pour obtenir des recommandations basées sur un titre de film
# @app.route('/recommend', methods=['GET'])
# def get_recommendations():
#     movie_title = request.args.get('title')
#     if not movie_title:
#         return jsonify({"error": "Le paramètre 'title' est requis"}), 400

#     try:
#         # Étape 1 : Trouver le movieId dans l'index "movies"
#         logger.debug(f"Recherche du film : {movie_title}")
#         movie_query = {
#             "query": {
#                 "match": {
#                     "title": movie_title
#                 }
#             }
#         }
#         movie_response = es.search(index="movies", body=movie_query)
#         movie_hits = movie_response["hits"]["hits"]

#         if not movie_hits:
#             logger.warning(f"Aucun film trouvé pour : {movie_title}")
#             return jsonify({"error": "Aucun film trouvé avec ce titre dans l'index 'movies'"}), 404

#         movie_id = movie_hits[0]["_source"]["movieId"]
#         logger.info(f"Film trouvé : {movie_title}, movieId : {movie_id}")

#         # Étape 2 : Trouver les utilisateurs ayant interagi avec ce film dans "ratings"
#         user_query = {
#             "query": {
#                 "term": {
#                     "movieId": movie_id
#                 }
#             }
#         }
#         user_response = es.search(index="ratings", body=user_query)
#         user_hits = user_response["hits"]["hits"]

#         if not user_hits:
#             logger.info(f"Aucun utilisateur trouvé pour movieId : {movie_id} dans 'ratings'")
#             return jsonify({"message": "Aucun utilisateur trouvé pour ce film dans les interactions"}), 200

#         # Récupérer les userId associés
#         user_ids = [hit["_source"]["userId"] for hit in user_hits]
#         logger.debug(f"Utilisateurs trouvés : {user_ids}")

#         # Étape 3 : Obtenir les recommandations pour ces utilisateurs dans "recommendations"
#         recommendation_query = {
#             "query": {
#                 "terms": {
#                     "userId": user_ids
#                 }
#             },
#             "aggs": {
#                 "by_movie": {
#                     "terms": {
#                         "field": "recommendations.movieId",  # Champ imbriqué dans "recommendations"
#                         "size": 10  # Top 10 recommandations
#                     }
#                 }
#             }
#         }
#         recommendation_response = es.search(index="recommendations", body=recommendation_query)
        
#         # Récupérer les movieId recommandés
#         movie_ids = [bucket["key"] for bucket in recommendation_response["aggregations"]["by_movie"]["buckets"]]

#         # Étape 4 : Récupérer les titres des films recommandés dans "movies"
#         movies_query = {
#             "query": {
#                 "terms": {
#                     "movieId": movie_ids
#                 }
#             }
#         }
#         movies_response = es.search(index="movies", body=movies_query)
#         recommended_movies = [
#             {"movieId": hit["_source"]["movieId"], "title": hit["_source"]["title"]}
#             for hit in movies_response["hits"]["hits"]
#         ]

#         return jsonify({
#             "movie_title": movie_title,
#             "recommendations": recommended_movies
#         })

#     except Exception as e:
#         logger.error(f"Erreur : {str(e)}")
#         return jsonify({"error": str(e)}), 500

# # Route de test
# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({"message": "API de recommandation active !"})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_ENV") == "development")
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import logging
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)

es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
es = Elasticsearch([es_url])

# Charger le modèle
model_path = os.getenv("MODEL_PATH", "/opt/spark_jobs/systeme/genre_predictor.pkl")
model = joblib.load(model_path)
logger.info(f"Modèle chargé depuis {model_path}")

if not es.ping():
    raise ValueError("Impossible de se connecter à Elasticsearch !")
logger.info("Connexion à Elasticsearch réussie.")

# Liste des genres
genres = ["action", "adventure", "animation", "children", "comedy", "crime",
          "documentary", "drama", "fantasy", "filmnoir", "horror", "musical",
          "mystery", "romance", "scifi", "thriller", "war", "western"]

@app.route('/recommend_session', methods=['GET'])
def recommend_session():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"error": "Le paramètre 'userId' est requis"}), 400

    try:
        user_id = int(user_id)

        # Vérifier si l'utilisateur existe dans "user_profiles"
        user_query = {
            "query": {
                "term": {
                    "userId": user_id
                }
            }
        }
        user_response = es.search(index="user_profiles", body=user_query)
        user_hits = user_response["hits"]["hits"]

        if user_hits:
            # Utilisateur existant
            logger.info(f"Utilisateur {user_id} existant.")
            user_profile = user_hits[0]["_source"]
            
            functions = []
            for genre in genres:
                avg_key = f"avg_{genre}"
                if avg_key in user_profile and user_profile[avg_key]:
                    functions.append({
                        "filter": {"term": {genre.capitalize(): 1}},
                        "weight": float(user_profile[avg_key])
                    })

            movies_query = {
                "query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "functions": functions,
                        "score_mode": "sum",
                        "boost_mode": "sum"
                    }
                },
                "size": 10
            }
        else:
            # Nouvel utilisateur : prédiction avec le modèle
            # Dans la partie "else" pour les nouveaux utilisateurs :
            logger.info(f"Utilisateur {user_id} nouveau.")
            gender = request.args.get('gender')
            occupation = request.args.get('occupation')
            age = request.args.get('age')  # Nouvelle entrée
            if not gender or not occupation or not age:
                return jsonify({"error": "Pour un nouvel utilisateur, 'gender', 'occupation' et 'age' sont requis"}), 400

            input_data = pd.DataFrame({
                "age": [int(age)],
                "gender": [gender.upper()],
                "occupation": [occupation.lower()]
            })
            X_new = pd.get_dummies(input_data, drop_first=True)

            # Alignement des colonnes avec celles du modèle
            model_features = model.feature_names_in_  # Colonnes vues pendant l'entraînement
            X_new = X_new.reindex(columns=model_features, fill_value=0)

            # Prédire les préférences par genre
            predicted_scores = model.predict(X_new)[0]
            genre_scores = dict(zip(genres, predicted_scores))

            # Construire la requête Elasticsearch
            functions = [
                {"filter": {"term": {genre.capitalize(): 1}}, "weight": float(score)}
                for genre, score in genre_scores.items() if score > 0
            ]
            movies_query = {
                "query": {
                    "function_score": {
                        "query": {"match_all": {}},
                        "functions": functions,
                        "score_mode": "sum",
                        "boost_mode": "sum"
                    }
                },
                "size": 10
            }

        # Exécuter la requête
        movies_response = es.search(index="movies", body=movies_query)
        recommendations = [
            {"movieId": hit["_source"]["movieId"], "title": hit["_source"]["title"]}
            for hit in movies_response["hits"]["hits"]
        ]

        return jsonify({
            "userId": user_id,
            "recommendations": recommendations
        })

    except Exception as e:
        logger.error(f"Erreur : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def ui():
    return '''
        <html>
        <head>
            <title>Système de Recommandation de Films</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .section { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                h2 { color: #2c3e50; }
                form { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; }
                input, select { width: 100%; padding: 8px; margin-bottom: 10px; }
                button { padding: 10px 20px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #2980b9; }
                .results { margin-top: 20px; }
                .movie { padding: 10px; border-bottom: 1px solid #eee; }
            </style>
        </head>
        <body>
            <h1>Système de Recommandation de Films</h1>
            
            <!-- Section pour les utilisateurs existants -->
            <div class="section">
                <h2>Recommandations pour un utilisateur existant</h2>
                <form action="/recommend_session" method="get" target="_blank">
                    <label>ID utilisateur (ex: 1, 2, 3):</label>
                    <input type="number" name="userId" required>
                    <button type="submit">Obtenir les recommandations</button>
                </form>
            </div>

            <!-- Section pour les nouveaux utilisateurs -->
            <div class="section">
                <h2>Recommandations pour un nouvel utilisateur</h2>
                <form action="/recommend_session" method="get" target="_blank">
                    <label>Genre (M/F):</label>
                    <input type="text" name="gender" placeholder="M ou F" required>
                    
                    <label>Âge:</label>
                    <input type="number" name="age" placeholder="Entrez votre âge" required>
                    
                    <label>Occupation:</label>
                    <select name="occupation" required>
                        <option value="">Sélectionnez une occupation</option>
                        <option value="academic/educator">Académique/Éducateur</option>
                        <option value="artist">Artiste</option>
                        <option value="clerical/admin">Administratif</option>
                        <option value="college/grad student">Étudiant</option>
                        <option value="customer service">Service client</option>
                        <option value="doctor/health care">Médecin/Santé</option>
                        <option value="executive/managerial">Exécutif/Manager</option>
                        <option value="farmer">Agriculteur</option>
                        <option value="homemaker">Maison</option>
                        <option value="lawyer">Avocat</option>
                        <option value="librarian">Bibliothécaire</option>
                        <option value="programmer">Programmeur</option>
                        <option value="retired">Retraité</option>
                        <option value="sales/marketing">Vente/Marketing</option>
                        <option value="scientist">Scientifique</option>
                        <option value="student">Étudiant</option>
                        <option value="technician/engineer">Technicien/Ingénieur</option>
                        <option value="tradesman/craftsman">Artisan</option>
                        <option value="unemployed">Sans emploi</option>
                        <option value="writer">Écrivain</option>
                    </select>
                    <button type="submit">Obtenir les recommandations</button>
                </form>
            </div>

            <!-- Section pour les recommandations basées sur un film -->
            <div class="section">
                <h2>Recommandations basées sur un film</h2>
                <form action="/recommend" method="get" target="_blank">
                    <label>Titre du film (ex: "Toy Story", "Jumanji"):</label>
                    <input type="text" name="title" required>
                    <button type="submit">Obtenir les recommandations</button>
                </form>
            </div>
        </body>
        </html>
    '''
@app.route('/results')
def results():
    recommendations = request.args.getlist('recommendations')
    user_id = request.args.get('userId')
    movie_title = request.args.get('movie_title')
    
    return f'''
        <html>
        <head>
            <title>Résultats des Recommandations</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .recommendation {{ margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                h2 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>Résultats des Recommandations</h1>
            {f"<h2>Pour l'utilisateur {user_id}</h2>" if user_id else ""}
            {f"<h2>Basé sur le film '{movie_title}'</h2>" if movie_title else ""}
            <div class="recommendations">
                {"".join([f'<div class="recommendation"><strong>{rec["title"]}</strong> (ID: {rec["movieId"]})</div>' for rec in recommendations])}
            </div>
            <a href="/ui">Retour à la page principale</a>
        </body>
        </html>
    '''

def home():
    return jsonify({"message": "API de recommandation par profil active !"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_ENV") == "development")