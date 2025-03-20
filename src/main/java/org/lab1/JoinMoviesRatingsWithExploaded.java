package org.lab1;

import org.apache.spark.sql.*;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;

import java.io.IOException;
import java.util.List;

public class JoinMoviesRatingsWithExploaded {

    static void plot_histogram(List<Double> x, List<Double> weights, String title) {
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        
        plt.hist().add(x).weights(weights).bins(50);
        plt.title(title);
        try {
                plt.show();
        } catch (IOException e) {
                throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
                throw new RuntimeException(e);
        }
    }
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
                .load("src/main/resources/lab1/ratings.csv");
        //
        Dataset<Row> df_movies = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/lab1/movies.csv")
                .withColumn("genres_array", split(col("genres"), "\\|")) // Extracts the year part
                .drop("genres")
                .withColumn("genre", explode(col("genres_array"))) // Extracts the year part
                .drop("genres_array")
                .withColumn("title2",
                when(regexp_extract(col("title"),"^(.*?)\\s*\\((\\d{4})\\)\\s*$",1).
                        equalTo(""), col("title"))
                        .otherwise(regexp_extract(col("title"),"^(.*?)\\s*\\((\\d{4})\\)\\s*$",1)))                
                .withColumn("year", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)\\s*$", 2)) // Poprawione wyrażenie
                .withColumn("title2", when(col("year").equalTo(""), col("title")).otherwise(col("title2"))) // Obsługa braku daty
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
        Helper.plot_histogram(df_stats_ym, "Rozktad róznicy lat pomiedzy ocena a wydaniem filmu");

        
        
        var df_mr2 = df_mr.groupBy("release_to_rating_year")
        .count()
        .orderBy("release_to_rating_year");

        df_mr2.show(20);
        df_mr2.filter("release_to_rating_year=-1 OR release_to_rating_year IS NULL").show(105);

        var df_mr2_histogram = df_mr2.filter("release_to_rating_year!=-1 AND release_to_rating_year IS NOT NULL");

        plot_histogram(df_mr2_histogram.select("release_to_rating_year").as(Encoders.DOUBLE()).collectAsList(),
                df_mr2_histogram.select("count").as(Encoders.DOUBLE()).collectAsList(), "[PO ZMIANACH] Rozktad róznicy lat pomiedzy ocena a wydaniem filmu");
    }
}
