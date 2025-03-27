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

public class LinearRegressionPolynomialFeaturesOrderTwo {
    
    static void processDataset(SparkSession spark, String filename, Function<Double, Double> f_true,
            double regParam, double elasticNetParam, int maxIter) {
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/lab2/" + filename);

        Dataset<Row> dataWithX2 = data.withColumn("X2", functions.col("X").multiply(functions.col("X")));

        System.out.println("Data with polynomial features:");
        dataWithX2.show(5);
        dataWithX2.printSchema();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] { "X", "X2" })
                .setOutputCol("features");

        Dataset<Row> vectorData = vectorAssembler.transform(dataWithX2);

        List<Row> rows = data.collectAsList();
        List<Double> xValues = rows.stream()
                .map(row -> row.getDouble(0))
                .collect(Collectors.toList());
        List<Double> yValues = rows.stream()
                .map(row -> row.getDouble(1))
                .collect(Collectors.toList());

        Model.trainAndEvaluate(vectorData, regParam, elasticNetParam, maxIter, xValues, yValues, f_true, filename, 2);
    }

    public static void main(String[] args) {
        var resourceName = args[0];
        System.out.println("Resource name: " + resourceName);

        SparkSession spark = SparkSession.builder()
                .appName("Polynomial Features Order 2")
                .master("local[*]")
                .getOrCreate();

        processDataset(spark, resourceName, null, 10, 0.8, 100);

        spark.stop();
    }
}
