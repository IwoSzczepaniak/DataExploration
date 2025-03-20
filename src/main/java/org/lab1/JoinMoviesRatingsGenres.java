package org.lab1;

import org.apache.spark.sql.*;

import static org.apache.spark.sql.functions.*;

public class JoinMoviesRatingsGenres {
        public static void main(String[] args) {
                SparkSession spark = SparkSession.builder()
                                .appName("JoinMoviesRatingsGenres")
                                .master("local")
                                .getOrCreate();
                System.out.println("Using Apache Spark v" + spark.version());

                Dataset<Row> df_ratings = spark.read()
                                .format("csv")
                                .option("header", "true")
                                .option("inferSchema", "true")
                                .load("src/main/resources/lab1/ratings.csv");

                Dataset<Row> df_movies = spark.read()
                                .format("csv")
                                .option("header", "true")
                                .option("inferSchema", "true")
                                .load("src/main/resources/lab1/movies.csv")
                                .withColumn("genres_array", split(col("genres"), "\\|"))
                                .drop("genres")
                                .withColumn("genre", explode(col("genres_array")))
                                .drop("genres_array")
                                .withColumn("title2",
                                                regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)\\s*$", 1))
                                .withColumn("year", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)\\s*$", 2))
                                .withColumn("title2",
                                                when(col("year").equalTo(""), col("title")).otherwise(col("title2")))
                                .drop("title")
                                .withColumnRenamed("title2", "title");

                var df_mr = df_movies.join(df_ratings, "movieId", "inner");

                df_mr = df_mr
                                .withColumn("datetime", functions.from_unixtime(df_mr.col("timestamp")))
                                .drop("timestamp");

                df_mr.show(10);
                System.out.println("Dataframe's schema:");
                df_mr.printSchema();

                var df_genre_stats = df_mr.groupBy("genre")
                                .agg(
                                                functions.min("rating").alias("min_rating"),
                                                functions.avg("rating").alias("avg_rating"),
                                                functions.max("rating").alias("max_rating"),
                                                functions.count("rating").alias("rating_cnt"))
                                .orderBy(functions.col("genre"));

                System.out.println("Genre statistics:");
                df_genre_stats.show(20);

                System.out.println("Top 3 genres by average rating:");
                df_genre_stats.orderBy(functions.col("avg_rating").desc())
                                .limit(3)
                                .show();

                System.out.println("Top 3 genres by number of ratings:");
                df_genre_stats.orderBy(functions.col("rating_cnt").desc())
                                .limit(3)
                                .show();

                double globalAvgRating = df_ratings.agg(functions.avg("rating")).first().getDouble(0);
                System.out.println("Średnia ocena dla całego zbioru: " + globalAvgRating);

                var df_genre_stats_filtered = df_genre_stats.filter(col("avg_rating").gt(globalAvgRating));

                System.out.println("Statystyki gatunków ze średnią oceną powyżej średniej globalnej:");
                df_genre_stats_filtered.orderBy(col("avg_rating").desc()).show();

                df_mr.createOrReplaceTempView("movies_ratings");
                df_ratings.createOrReplaceTempView("ratings");
                String query = """
                       SELECT genre, AVG(rating) AS avg_rating, COUNT(rating) 
                       FROM movies_ratings GROUP BY genre 
                       HAVING AVG(rating) > (SELECT AVG(rating) FROM ratings) 
                       ORDER BY avg_rating DESC""";
                var df_cat_above_avg = spark.sql(query);
                df_cat_above_avg.show();
        }
}