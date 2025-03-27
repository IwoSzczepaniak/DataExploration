package org.lab3;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.lab2.Model;

public class LinearRegressionPolynomialFeaturesOrderThree {
    
    static void processDataset(SparkSession spark, String filename, Function<Double, Double> f_true,
            double regParam, double elasticNetParam, int maxIter) {
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/lab2/" + filename);

        Dataset<Row> dataWithPolynomialFeatures = data
                .withColumn("X2", functions.col("X").multiply(functions.col("X")))
                .withColumn("X3", functions.col("X").multiply(functions.col("X")).multiply(functions.col("X")));

        System.out.println("Data with polynomial features:");
        dataWithPolynomialFeatures.show(5);
        dataWithPolynomialFeatures.printSchema();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] { "X", "X2", "X3" })
                .setOutputCol("features");

        Dataset<Row> vectorData = vectorAssembler.transform(dataWithPolynomialFeatures);

        List<Row> rows = data.collectAsList();
        List<Double> xValues = rows.stream()
                .map(row -> row.getDouble(0))
                .collect(Collectors.toList());
        List<Double> yValues = rows.stream()
                .map(row -> row.getDouble(1))
                .collect(Collectors.toList());

        Model.trainAndEvaluate(vectorData, regParam, elasticNetParam, maxIter, xValues, yValues, f_true, filename, 3);
    }

    public static void main(String[] args) {
        var resourceName = args[0];
        System.out.println("Resource name: " + resourceName);

        SparkSession spark = SparkSession.builder()
                .appName("Polynomial Features Order 3")
                .master("local[*]")
                .getOrCreate();

        processDataset(spark, resourceName, null, 0.3, 0.8, 10);

        spark.stop();
    }
} 