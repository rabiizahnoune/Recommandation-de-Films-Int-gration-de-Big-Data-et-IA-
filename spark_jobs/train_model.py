from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os

# Initialiser Spark
spark = SparkSession.builder \
    .appName("MovieLens_recommendation") \
    .config("spark.master", "spark://spark-master:7077") \
    .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.5.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

# Charger les données ratings
data_path = "/opt/spark_jobs/data/raw/ml-100k"
ratings_file = os.path.join(data_path, "u.data")

if not os.path.exists(ratings_file):
    raise FileNotFoundError(f"File {ratings_file} not found!")

ratings_df = spark.read.option("delimiter", "\t").csv(ratings_file, header=False, inferSchema=True) \
    .toDF("userId", "movieId", "rating", "timestamp") \
    .dropDuplicates(["userId", "movieId"]) \
    .na.drop(subset=["rating"])

# Diviser les données en ensembles d'entraînement et de test
(training_df, test_df) = ratings_df.randomSplit([0.8, 0.2], seed=42)

# Configurer le modèle ALS
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)

# Entraîner le modèle
model = als.fit(training_df)

# Faire des prédictions sur l'ensemble de test
predictions = model.transform(test_df)

# Évaluer le modèle
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) = {rmse}")

# Générer des recommandations pour tous les utilisateurs (top 10 par utilisateur)
user_recommendations = model.recommendForAllUsers(10)
user_recommendations.show(truncate=False)

# Sauvegarder les recommandations dans Elasticsearch
user_recommendations.write \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "elasticsearch") \
    .option("es.port", "9200") \
    .option("es.resource", "recommendations") \
    .option("es.write.operation", "index") \
    .option("es.nodes.wan.only", "true") \
    .mode("overwrite") \
    .save()

# Arrêter Spark
spark.stop()