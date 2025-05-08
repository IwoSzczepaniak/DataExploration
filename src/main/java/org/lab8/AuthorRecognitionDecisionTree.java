package org.lab8;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;

public class AuthorRecognitionDecisionTree {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Author Recognition using Decision Tree")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .option("delimiter", ",")
                .option("quote", "'")
                .option("inferSchema", "true")
                .load("src/main/resources/books/two-books-all-1000-10-stem.csv");

        System.out.println("Distinct authors and their works:");
        df.select("author", "work").distinct().show();

        System.out.println("\nNumber of documents per author:");
        Dataset<Row> authorCounts = df.groupBy("author").count();
        authorCounts.show();

        System.out.println("\nAverage text length per author:");
        Dataset<Row> avgLengths = df.withColumn("content_length", length(col("content")))
                .groupBy("author")
                .agg(avg("content_length").alias("avg_text_length"));
        avgLengths.show();

        spark.stop();
    }
} 