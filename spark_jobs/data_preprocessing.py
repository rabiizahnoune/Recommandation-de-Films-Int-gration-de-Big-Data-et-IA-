from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split 
import os 
import requests
import time

####################################################################################################################################
# Initialiser Spark
spark = SparkSession.builder \
    .appName("MovieLens_preprocessing") \
    .config("spark.master", "spark://spark-master:7077") \
    .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.5.1") \
    .getOrCreate()

# Enable logging for debugging
spark.sparkContext.setLogLevel("INFO")

# Vérifier la connexion à Elasticsearch
def check_elasticsearch_connection():
    es_url = "http://elasticsearch:9200"
    max_retries = 5
    retry_delay = 5  # secondes
    
    for attempt in range(max_retries):
        try:
            response = requests.get(es_url, timeout=10)
            if response.status_code == 200:
                print(f"Successfully connected to Elasticsearch at {es_url}")
                print(f"Elasticsearch response: {response.json()}")
                return True
            else:
                print(f"Unexpected status code {response.status_code} from Elasticsearch")
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Failed to connect to Elasticsearch at {es_url} - {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    raise ConnectionError(f"Failed to connect to Elasticsearch at {es_url} after {max_retries} attempts")

# Exécuter la vérification
check_elasticsearch_connection()

# Chemins des fichiers 
data_path = "/opt/spark_jobs/data/raw/ml-100k"
ratings_file = os.path.join(data_path, "u.data")
movies_file = os.path.join(data_path, "u.item")

# Vérifier si les fichiers existent
print(f"Checking if {ratings_file} exists...")
if not os.path.exists(ratings_file.replace("file://", "")):
    raise FileNotFoundError(f"File {ratings_file} not found!")
print(f"Checking if {movies_file} exists...")
if not os.path.exists(movies_file.replace("file://", "")):
    raise FileNotFoundError(f"File {movies_file} not found!")

# 1. Charger les données
ratings_df = spark.read.option("delimiter", "\t").csv(ratings_file, header=False, inferSchema=True) \
    .toDF("userId", "movieId", "rating", "timestamp")

movies_df = spark.read.option("delimiter", "|").csv(movies_file, header=False, inferSchema=True) \
    .toDF("movieId", "title", "release_date", "video_release_date", "imdb_url", "unknown", "action", 
          "adventure", "animation", "children", "comedy", "crime", "documentary", "drama", "fantasy", 
          "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western")

# 2. Nettoyage
ratings_df = ratings_df.dropDuplicates(["userId", "movieId"]).na.drop(subset=["rating"])
movies_df = movies_df.dropDuplicates(["movieId"])

# 3. Transformation
avg_ratings_df = ratings_df.groupBy("movieId").agg({"rating": "avg"}).withColumnRenamed("avg(rating)", "avg_rating")
avg_ratings_df.show()
processed_df = movies_df.join(avg_ratings_df, "movieId", "left") \
    .select("movieId", "title", "avg_rating", "action", "adventure", "animation", "children", "comedy", 
            "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", 
            "romance", "sci_fi", "thriller", "war", "western")

# 4. Sauvegarder dans Elasticsearch
processed_df.write \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "elasticsearch") \
    .option("es.port", "9200") \
    .option("es.resource", "movies") \
    .option("es.write.operation", "index") \
    .option("es.nodes.wan.only", "true") \
    .mode("overwrite") \
    .save()
# Aperçu
processed_df.show(10, truncate=False)

# Arrêter Spark
spark.stop()