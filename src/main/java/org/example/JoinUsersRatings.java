package org.example;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.count;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

public class JoinUsersRatings {
    private static void plotUserStats(List<Double> avgRatings, List<Double> counts) {
        // Konfiguracja Python
        PythonConfig pythonConfig = PythonConfig.pythonBinPathConfig("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3");
        
        // Wykres punktowy
        Plot plt = Plot.create(pythonConfig);
        plt.plot()
                .add(avgRatings, counts, "o")
                .label("Użytkownicy");

        plt.title("Średnia ocena vs Liczba ocen");
        plt.xlabel("Średnia ocena");
        plt.ylabel("Liczba ocen");
        plt.legend();
        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            e.printStackTrace();
        }

        // Histogram
        Plot pltHist = Plot.create(pythonConfig);
        pltHist.hist()
                .add(avgRatings)
                .bins(20)
                .label("Rozkład");

        pltHist.title("Rozkład średnich ocen");
        pltHist.xlabel("Średnia ocena");
        pltHist.ylabel("Liczba użytkowników");
        pltHist.legend();
        try {
            pltHist.show();
        } catch (IOException | PythonExecutionException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("JoinUsersRatings")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        Dataset<Row> df_users = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/users.csv");

        Dataset<Row> df_ratings = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/ratings.csv");

        Dataset<Row> df_ur = df_users.join(df_ratings, "userId");

        Dataset<Row> df_grouped = df_ur.groupBy("email")
                .agg(
                        avg("rating").alias("avg_rating"),
                        count("rating").alias("count"))
                .orderBy(col("avg_rating").desc());

        System.out.println("Statystyki ocen użytkowników:");
        df_grouped.show();

        // Przygotowanie danych do wykresu
        List<Row> plotData = df_grouped.collectAsList();
        List<Double> avgRatings = new ArrayList<>();
        List<Double> counts = new ArrayList<>();

        for (Row row : plotData) {
            avgRatings.add(row.getDouble(1)); // avg_rating
            counts.add((double) row.getLong(2)); // count
        }

        // Wywołanie funkcji plotującej
        plotUserStats(avgRatings, counts);
    }
}