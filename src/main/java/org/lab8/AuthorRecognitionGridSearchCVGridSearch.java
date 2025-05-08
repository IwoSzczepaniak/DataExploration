package org.lab8;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import java.util.ArrayList;
import java.util.List;

public class AuthorRecognitionGridSearchCVGridSearch {
    private static EvaluationResult performCV(SparkSession spark, String filename) {
        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .option("delimiter", ",")
                .option("quote", "'")
                .option("inferSchema", "true")
                .load(filename);

        Dataset<Row>[] splits = df.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> df_train = splits[0];
        Dataset<Row> df_test = splits[1];

        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern("[\\s\\p{Punct}\\u2014\\u2026\\u201C\\u201E]+");

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setMinDF(2)
                .setVocabSize(10_000);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini")
                .setMaxDepth(30);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, decisionTreeClassifier});

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(new ParamGridBuilder().build())
                .setNumFolds(3)
                .setParallelism(8);

        System.out.println("\nProcessing file: " + filename);
        System.out.println("Starting cross-validation...");
        CrossValidatorModel cvModel = cv.fit(df_train);

        Dataset<Row> predictions = cvModel.transform(df_test);
        
        double accuracy = 0, weightedPrecision = 0, weightedRecall = 0, f1 = 0;
        
        for (String metricName : new String[]{"accuracy", "weightedPrecision", "weightedRecall", "f1"}) {
            evaluator.setMetricName(metricName);
            double score = evaluator.evaluate(predictions);
            switch (metricName) {
                case "accuracy": accuracy = score; break;
                case "weightedPrecision": weightedPrecision = score; break;
                case "weightedRecall": weightedRecall = score; break;
                case "f1": f1 = score; break;
            }
        }

        return new EvaluationResult(filename, accuracy, weightedPrecision, weightedRecall, f1);
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Author Recognition using Decision Tree with CV")
                .master("local[*]")
                .getOrCreate();

        String[] filenames = {
                "src/main/resources/books/two-books-all-1000-1-stem.csv",
                "src/main/resources/books/two-books-all-1000-3-stem.csv",
                "src/main/resources/books/two-books-all-1000-5-stem.csv",
                "src/main/resources/books/two-books-all-1000-10-stem.csv",
                "src/main/resources/books/five-books-all-1000-1-stem.csv",
                "src/main/resources/books/five-books-all-1000-3-stem.csv",
                "src/main/resources/books/five-books-all-1000-5-stem.csv",
                "src/main/resources/books/five-books-all-1000-10-stem.csv"
        };

        List<EvaluationResult> results = new ArrayList<>();
        
        for (String filename : filenames) {
            results.add(performCV(spark, filename));
        }

        System.out.println("\nEvaluation Results:");
        System.out.println("------------------------------------------------------------------------------------------------------------");
        System.out.printf("%-45s | %-10s | %-10s | %-10s | %-10s%n", 
                "File", "Accuracy", "Precision", "Recall", "F1");
        System.out.println("------------------------------------------------------------------------------------------------------------");
        
        for (EvaluationResult result : results) {
            result.printResult();
        }
        System.out.println("------------------------------------------------------------------------------------------------------------");

        spark.stop();
    }
} 