package org.example;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.regexp_extract;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
public class JoinMoviesRatings {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("JoinMoviesRatings")
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
                .withColumn("title2", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 1)) // Extracts the title part
                .withColumn("year", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 2)) // Extracts the year part
                .drop("title") // Drops the original 'title' column
                .withColumnRenamed("title2", "title"); // Renames 'title2' to 'title';

        var df_mr = df_movies.join(df_ratings, "movieId", "inner");

        df_mr = df_mr.
                withColumn(
                        "datetime",
                        functions.from_unixtime(df_mr.col("timestamp"))).
                drop("timestamp");

        df_mr.show(20);
        System.out.println("Dataframe's schema:");
        df_mr.printSchema();

        // Zgrupuj dane po tytule używając funkcji groupBy() i dodając kolumny:
        //
        //z minimalną oceną - nazwa kolumny min_rating
        //średnią ocen o nazwie avg_rating
        //maksymalną oceną - nazwa kolumny max_rating
        //liczbą ocen - nazwa kolumny rating_cnt
        //Użyj funkcji agg( min(“rating”).alias(“min_rating”), …kolejna agregacja,…). Funkcja alias() służy do zmiany nazwy
        //
        //Posortuj po liczbie ocen (malejąco).
        var df_mr_t = df_mr.groupBy("title")
                .agg(
                        functions.min("rating").alias("min_rating"),
                        functions.avg("rating").alias("avg_rating"),
                        functions.max("rating").alias("max_rating"),
                        functions.count("rating").alias("rating_cnt")
                )
                .orderBy(functions.col("rating_cnt").desc());


        df_mr_t.show(20);
        System.out.println("Dataframe's schema:");
        df_mr_t.printSchema();

        var avgRatings = df_mr_t.select("avg_rating").where("rating_cnt>=0").as(Encoders.DOUBLE()).collectAsList();
        Helper.plot_histogram(avgRatings, "Średnie wartości ocen");


        var countWithHighRating = df_mr_t
                .select("rating_cnt")
                .where("avg_rating >= 4.5")
                .as(Encoders.DOUBLE())
                .collectAsList();
        Helper.plot_histogram(countWithHighRating, "Rozkład liczby ocen spełniajacych predykat avg_rating>=4.5");

        var conjunction = df_mr_t
                .select("rating_cnt")
                .where("avg_rating >= 3.5").where("rating_cnt>=20")
                .as(Encoders.DOUBLE())
                .collectAsList();
        Helper.plot_histogram(conjunction, "Rozkład liczby ocen spełniajacych predykat avg_rating>=3.5 and rating_cnt>20");
    }
}
