from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

spark = SparkSession.builder \
    .appName("MovieLens_genre_based_model") \
    .config("spark.master", "spark://spark-master:7077") \
    .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.5.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

data_path = "/opt/spark_jobs/data/raw/ml-100k"
ratings_file = os.path.join(data_path, "u.data")
movies_file = os.path.join(data_path, "u.item")
users_file = os.path.join(data_path, "u.user")
model_output = "/opt/spark_jobs/systeme/genre_predictor.pkl"

ratings_df = spark.read.option("delimiter", "\t").csv(ratings_file, header=False, inferSchema=True) \
    .toDF("userId", "movieId", "rating", "timestamp") \
    .dropDuplicates(["userId", "movieId"]) \
    .na.drop(subset=["rating"])

movies_df = spark.read.option("delimiter", "|").csv(movies_file, header=False, inferSchema=True) \
    .toDF("movieId", "title", "release_date", "video_release_date", "IMDb_URL",
          "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western") \
    .select("movieId", "title",
            *[col(c).cast("int") for c in ["unknown", "Action", "Adventure", "Animation", "Children",
                                           "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                                           "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                                           "Sci-Fi", "Thriller", "War", "Western"]])

users_df = spark.read.option("delimiter", "|").csv(users_file, header=False, inferSchema=True) \
    .toDF("userId", "age", "gender", "occupation", "zipcode")

user_movie_df = ratings_df.join(movies_df, "movieId", "inner") \
    .join(users_df, "userId", "inner")

user_profile_df = user_movie_df.groupBy("userId", "age", "gender", "occupation") \
    .agg(
        avg(when(col("Action") == 1, col("rating")).otherwise(None)).alias("avg_action"),
        avg(when(col("Adventure") == 1, col("rating")).otherwise(None)).alias("avg_adventure"),
        avg(when(col("Animation") == 1, col("rating")).otherwise(None)).alias("avg_animation"),
        avg(when(col("Children") == 1, col("rating")).otherwise(None)).alias("avg_children"),
        avg(when(col("Comedy") == 1, col("rating")).otherwise(None)).alias("avg_comedy"),
        avg(when(col("Crime") == 1, col("rating")).otherwise(None)).alias("avg_crime"),
        avg(when(col("Documentary") == 1, col("rating")).otherwise(None)).alias("avg_documentary"),
        avg(when(col("Drama") == 1, col("rating")).otherwise(None)).alias("avg_drama"),
        avg(when(col("Fantasy") == 1, col("rating")).otherwise(None)).alias("avg_fantasy"),
        avg(when(col("Film-Noir") == 1, col("rating")).otherwise(None)).alias("avg_filmnoir"),
        avg(when(col("Horror") == 1, col("rating")).otherwise(None)).alias("avg_horror"),
        avg(when(col("Musical") == 1, col("rating")).otherwise(None)).alias("avg_musical"),
        avg(when(col("Mystery") == 1, col("rating")).otherwise(None)).alias("avg_mystery"),
        avg(when(col("Romance") == 1, col("rating")).otherwise(None)).alias("avg_romance"),
        avg(when(col("Sci-Fi") == 1, col("rating")).otherwise(None)).alias("avg_scifi"),
        avg(when(col("Thriller") == 1, col("rating")).otherwise(None)).alias("avg_thriller"),
        avg(when(col("War") == 1, col("rating")).otherwise(None)).alias("avg_war"),
        avg(when(col("Western") == 1, col("rating")).otherwise(None)).alias("avg_western")
    )

user_profile_df.write \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "elasticsearch") \
    .option("es.port", "9200") \
    .option("es.resource", "user_profiles") \
    .option("es.write.operation", "index") \
    .option("es.nodes.wan.only", "true") \
    .mode("overwrite") \
    .save()

movies_df.write \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes", "elasticsearch") \
    .option("es.port", "9200") \
    .option("es.resource", "movies") \
    .option("es.write.operation", "index") \
    .option("es.nodes.wan.only", "true") \
    .mode("overwrite") \
    .save()

# Entraîner le modèle avec age
pandas_df = user_profile_df.toPandas()
X = pd.get_dummies(pandas_df[['age', 'gender', 'occupation']], drop_first=True)
y = pandas_df[[f"avg_{genre}" for genre in ["action", "adventure", "animation", "children", "comedy",
                                            "crime", "documentary", "drama", "fantasy", "filmnoir",
                                            "horror", "musical", "mystery", "romance", "scifi",
                                            "thriller", "war", "western"]]].fillna(0)

model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=5)
model.fit(X, y)
joblib.dump(model, model_output)
print(f"Modèle sauvegardé dans {model_output}")

spark.stop()