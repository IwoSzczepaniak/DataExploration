package org.lab8;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.RegexTokenizer;
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

        System.out.println("\nTokenized content sample:");
        String sep = "[\\s\\p{Punct}\\u2014\\u2026\\u201C\\u201E]+";
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern(sep);
        Dataset<Row> df_tokenized = tokenizer.transform(df);
        df_tokenized.show(3, true);

        spark.stop();
    }
} 