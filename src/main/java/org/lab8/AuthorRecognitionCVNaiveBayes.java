package org.lab8;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class AuthorRecognitionCVNaiveBayes {
    private static void performCV(SparkSession spark, String filename, int vocabSize, String modelType) {
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
                .setVocabSize(vocabSize);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        NaiveBayes nb = new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setSmoothing(0.2)
                .setModelType(modelType);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, countVectorizer, labelIndexer, nb});

        ParamMap[] paramGrid = new ParamGridBuilder().build();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator.setMetricName("f1"))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3)
                .setParallelism(8);

        CrossValidatorModel cvModel = cv.fit(df_train);

        double cvScore = cvModel.avgMetrics()[0];

        Dataset<Row> predictions = cvModel.transform(df_test);
        
        evaluator.setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        
        evaluator.setMetricName("weightedPrecision");
        double precision = evaluator.evaluate(predictions);
        
        evaluator.setMetricName("weightedRecall");
        double recall = evaluator.evaluate(predictions);
        
        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);

        System.out.printf("| %-35s | %8.4f | %8.4f |  %8.4f | %8.4f | %8.4f |%n",
                filename.substring(filename.lastIndexOf("/") + 1),
                cvScore, accuracy, precision, recall, f1);
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Author Recognition using Naive Bayes with CV")
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

        System.out.println("\n| Dataset                             |    CV F1 | Accuracy | Precision |   Recall |       F1 |");
        System.out.println("|-------------------------------------|----------|----------|-----------|----------|----------|");

        for (String filename : filenames) {
            performCV(spark, filename, 5000, "multinomial");
        }

        spark.stop();
    }
} 