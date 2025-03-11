import requests

# Fonction pour vérifier la connexion à Elasticsearch
def check_elasticsearch_connection(host, port):
    url = f"http://{host}:{port}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Connexion à Elasticsearch établie avec succès.")
            return True
        else:
            print(f"Impossible de se connecter à Elasticsearch. Code de statut : {response.status_code}")
            return False
    except requests.ConnectionError:
        print("Impossible de se connecter à Elasticsearch. Vérifiez l'URL et le port.")
        return False

# Fonction pour vérifier si l'index existe
def check_index_exists(host, port, index_name):
    url = f"http://{host}:{port}/{index_name}"
    try:
        response = requests.head(url)
        if response.status_code == 200:
            print(f"L'index '{index_name}' existe.")
            return True
        else:
            print(f"L'index '{index_name}' n'existe pas. Code de statut : {response.status_code}")
            return False
    except requests.ConnectionError:
        print("Impossible de se connecter à Elasticsearch pour vérifier l'index.")
        return False

# Fonction pour vérifier le nombre de documents dans l'index
def check_documents_count(host, port, index_name):
    url = f"http://{host}:{port}/{index_name}/_count"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            count = response.json()["count"]
            print(f"Nombre de documents dans l'index '{index_name}' : {count}")
            return count > 0
        else:
            print(f"Impossible de compter les documents. Code de statut : {response.status_code}")
            return False
    except Exception as e:
        print(f"Erreur lors du comptage des documents : {str(e)}")
        return False


# Ajoutez les vérifications ici
import requests
import time

# [Ajoutez les fonctions check_elasticsearch_connection, check_index_exists et check_documents_count ici]

# Paramètres de connexion à Elasticsearch
es_host = "elasticsearch"
es_port = "9200"
index_name = "movies"

# Vérification de la connexion à Elasticsearch
if not check_elasticsearch_connection(es_host, es_port):
    raise Exception("Impossible de se connecter à Elasticsearch. Arrêt du script.")

# Attente pour permettre à Elasticsearch de traiter les écritures
time.sleep(10)

# Vérification de l'index
if not check_index_exists(es_host, es_port, index_name):
    raise Exception(f"L'index '{index_name}' n'a pas été créé correctement.")

# Vérification du nombre de documents
if not check_documents_count(es_host, es_port, index_name):
    raise Exception("Aucun document trouvé dans l'index. La sauvegarde a échoué.")

print("Vérifications réussies. Les données ont été correctement sauvegardées dans Elasticsearch.")