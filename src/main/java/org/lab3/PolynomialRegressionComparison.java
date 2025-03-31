package org.lab3;

import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class PolynomialRegressionComparison {
    
    static class RegressionResult {
        double r2;
        double mse;
        
        RegressionResult(double r2, double mse) {
            this.r2 = r2;
            this.mse = mse;
        }
    }
    
    static Map<String, Map<Integer, RegressionResult>> results = new TreeMap<>();
    
    static void processDataset(SparkSession spark, String filename, int order, double regParam, double elasticNetParam, int maxIter) {
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/lab2/" + filename);
        
        Dataset<Row> dataWithFeatures;
        if (order == 2) {
            dataWithFeatures = data.withColumn("X2", functions.col("X").multiply(functions.col("X")));
        } else {
            dataWithFeatures = data
                    .withColumn("X2", functions.col("X").multiply(functions.col("X")))
                    .withColumn("X3", functions.col("X").multiply(functions.col("X")).multiply(functions.col("X")));
        }
        
        String[] featureCols = new String[order];
        for (int i = 0; i < order; i++) {
            featureCols[i] = i == 0 ? "X" : "X" + (i + 1);
        }
        
        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");
        
        Dataset<Row> vectorData = vectorAssembler.transform(dataWithFeatures);
        
        LinearRegression lr = new LinearRegression()
                .setMaxIter(maxIter)
                .setRegParam(regParam)
                .setElasticNetParam(elasticNetParam)
                .setFeaturesCol("features")
                .setLabelCol("Y");
        
        LinearRegressionModel lrModel = lr.fit(vectorData);
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        
        results.computeIfAbsent(filename, k -> new TreeMap<>())
               .put(order, new RegressionResult(trainingSummary.r2(), trainingSummary.meanSquaredError()));
        
        System.out.printf("Coefficients for %s order %d: [%s]%n", 
            filename, 
            order,
            Arrays.stream(lrModel.coefficients().toArray())
                  .mapToObj(coeff -> String.format("%.3f", coeff))
                  .collect(Collectors.joining(",")));
    }
    
    static void printComparisonTable() {
        System.out.println("Dataset\t\t2nd Degree R²\t2nd Degree MSE\t3rd Degree R²\t3rd Degree MSE");
        System.out.println("--------\t-------------\t--------------\t-------------\t--------------");
        
        for (Map.Entry<String, Map<Integer, RegressionResult>> entry : results.entrySet()) {
            String dataset = entry.getKey();
            Map<Integer, RegressionResult> orderResults = entry.getValue();
            
            RegressionResult order2 = orderResults.get(2);
            RegressionResult order3 = orderResults.get(3);
            
            System.out.printf("%-16s| %-16.3f| %-16.3f| %-16.3f| %-16.3f%n",
                    dataset,
                    order2.r2,
                    order2.mse,
                    order3.r2,
                    order3.mse);
        }
    }
    
    public static void main(String[] args) {
        String[] resources = { "xy-001.csv", "xy-002.csv", "xy-003.csv", "xy-004.csv", "xy-005.csv", "xy-006.csv",
                "xy-007.csv", "xy-008.csv", "xy-009.csv", "xy-010.csv" };
        
        SparkSession spark = SparkSession.builder()
                .appName("Polynomial Regression Comparison")
                .master("local[*]")
                .getOrCreate();
        
        try {
            for (String resource : resources) {
                // Run 2nd degree regression
                processDataset(spark, resource, 2, 10.0, 0.8, 100);
                // Run 3rd degree regression
                processDataset(spark, resource, 3, 0.3, 0.8, 10);
                System.out.println("--------------------------------");
            }
            
            printComparisonTable();
        } finally {
            spark.stop();
        }
    }
} 