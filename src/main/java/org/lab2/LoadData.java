package org.lab2;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LoadData {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadMovies")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/lab2/xy-001.csv");

        System.out.println("Oryginalne dane:");
        data.show(5);
        data.printSchema();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"X"})
                .setOutputCol("features");

        Dataset<Row> vectorData = vectorAssembler.transform(data);

        System.out.println("\nDane po transformacji:");
        vectorData.show(5);
        vectorData.printSchema();

        spark.stop();
    }
}
