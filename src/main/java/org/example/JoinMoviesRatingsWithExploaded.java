package org.example;

import org.apache.spark.sql.*;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;

public class JoinMoviesRatingsWithExploaded {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("JoinMoviesRatingsWithExploaded")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        Dataset<Row> df_ratings = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/ratings.csv");
        //
        Dataset<Row> df_movies = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/movies.csv")
                .withColumn("genres_array", split(col("genres"), "\\|")) // Extracts the year part
                .drop("genres")
                .withColumn("genre", explode(col("genres_array"))) // Extracts the year part
                .drop("genres_array")
                .withColumn("title2", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 1)) // Extracts the title part
                .withColumn("year", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 2)) // Extracts the year part
                .drop("title") // Drops the original 'title' column
                .withColumnRenamed("title2", "title"); // Renames 'title2' to 'title';

        var df_mr = df_movies.join(df_ratings, "movieId", "inner");

        df_mr = df_mr.
                withColumn("datetime", functions.from_unixtime(df_mr.col("timestamp"))).
                drop("timestamp");

        df_mr = df_mr.
                withColumn(
                        "release_to_rating_year",
                        year(col("datetime")).minus(col("year"))
                );

        df_mr.show(20);
        System.out.println("Dataframe's schema:");
        df_mr.printSchema();


        var df_stats_ym = df_mr.select("release_to_rating_year").as(Encoders.DOUBLE()).sample(false, 0.002).collectAsList();
        Helper.plot_histogram(df_stats_ym, "release_to_rating_year");
    }
}
