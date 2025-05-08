package org.lab8;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.Vector;
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

        // Bag of Words conversion
        System.out.println("\nBag of Words transformation:");
        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)  // Set the maximum size of the vocabulary
                .setMinDF(2);          // Set the minimum number of documents in which a term must appear

        CountVectorizerModel countVectorizerModel = countVectorizer.fit(df_tokenized);
        Dataset<Row> df_bow = countVectorizerModel.transform(df_tokenized);

        System.out.println("\nSample of words and their feature vectors:");
        df_bow.select("words", "features").show(5, true);

        System.out.println("\nDetailed analysis of first document:");
        Row firstRow = df_bow.first();
        Vector features = (Vector) firstRow.get(df_bow.schema().fieldIndex("features"));
        String[] vocabulary = countVectorizerModel.vocabulary();

        System.out.println("First 20 word frequencies in first document:");
        int[] indices = features.toSparse().indices();
        double[] values = features.toSparse().values();
        for (int i = 0; i < Math.min(indices.length, 20); i++) {
            String word = vocabulary[indices[i]];
            double count = values[i];
            System.out.printf("%s -> %.6f%n", word, count);
        }

        System.out.println("\nConverting authors to numeric labels:");
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");
        StringIndexerModel labelModel = labelIndexer.fit(df_bow);
        df_bow = labelModel.transform(df_bow);
        df_bow.select("author", "label").distinct().show();

        System.out.println("\nTraining Decision Tree classifier:");
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini")
                .setMaxDepth(30);

        DecisionTreeClassificationModel model = dt.fit(df_bow);

        System.out.println("\nMaking predictions:");
        Dataset<Row> df_predictions = model.transform(df_bow);
        df_predictions.select("author", "label", "prediction").show();

        System.out.println("\nModel evaluation:");
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        for(String metric : new String[]{"f1", "accuracy"}){
            evaluator.setMetricName(metric);
            double score = evaluator.evaluate(df_predictions);
            System.out.printf("%s Score: %.4f%n", metric, score);
        }

        spark.stop();
    }
} 