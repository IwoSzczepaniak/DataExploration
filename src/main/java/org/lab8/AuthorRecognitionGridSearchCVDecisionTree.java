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
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;

public class AuthorRecognitionGridSearchCVDecisionTree {
    private static void performGridSearchCV(SparkSession spark, String filename) {
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
                .setMinDF(2);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, decisionTreeClassifier});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(countVectorizer.vocabSize(), new int[] {100, 1000, 10_000})
                .addGrid(decisionTreeClassifier.maxDepth(), new int[] {10, 20, 30})
                .build();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3)
                .setParallelism(8);

        System.out.println("Starting cross-validation...");
        CrossValidatorModel cvModel = cv.fit(df_train);

        System.out.println("\nBest model parameters:");
        PipelineModel bestModel = (PipelineModel) cvModel.bestModel();
        for (PipelineStage stage : bestModel.stages()) {
            System.out.println(stage);
        }

        System.out.println("\nAverage metrics for all parameter combinations:");
        double[] avgMetrics = cvModel.avgMetrics();
        for (int i = 0; i < avgMetrics.length; i++) {
            System.out.printf("Parameter set %d: F1 = %.4f%n", i + 1, avgMetrics[i]);
        }

        Dataset<Row> predictions = bestModel.transform(df_test);
        System.out.println("\nTest set evaluation metrics:");
        
        String[] metricNames = {"accuracy", "weightedPrecision", "weightedRecall", "f1"};
        for (String metricName : metricNames) {
            evaluator.setMetricName(metricName);
            double score = evaluator.evaluate(predictions);
            System.out.printf("%s: %.4f%n", metricName, score);
        }
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Author Recognition using Decision Tree with Grid Search CV")
                .master("local[*]")
                .getOrCreate();

        performGridSearchCV(spark, "src/main/resources/books/two-books-all-1000-10-stem.csv");

        spark.stop();
    }
} 