from pyspark.sql import SparkSession
import os

# Initialiser Spark
spark = SparkSession.builder \
    .appName("MovieLens_ratings_to_elasticsearch") \
    .config("spark.master", "spark://spark-master:7077") \
    .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.5.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

# Chemin vers les données MovieLens
data_path = "/opt/spark_jobs/data/raw/ml-100k"
ratings_file = os.path.join(data_path, "u.data")

# Vérifier si le fichier existe
if not os.path.exists(ratings_file):
    raise FileNotFoundError(f"File {ratings_file} not found!")

# Charger les données ratings depuis u.data
ratings_df = spark.read.option("delimiter", "\t").csv(ratings_file, header=False, inferSchema=True) \
    .toDF("userId", "movieId", "rating", "timestamp") \
    .dropDuplicates(["userId", "movieId"]) \
    .na.drop(subset=["rating"])

# Sauvegarder dans Elasticsearch dans l'index "ratings"
ratings_df.write \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "elasticsearch") \
    .option("es.port", "9200") \
    .option("es.resource", "ratings") \
    .option("es.write.operation", "index") \
    .option("es.nodes.wan.only", "true") \
    .mode("overwrite") \
    .save()

print("L'index 'ratings' a été ajouté à Elasticsearch avec succès.")

# Arrêter Spark
spark.stop()