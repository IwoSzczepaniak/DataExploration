package org.lab1;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class LoadUsers {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadUsers")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());
//
//        Dataset<Row> df = spark.read()
//                .format("csv")
//                .option("header", "true")
//                .load("src/main/resources/lab1/users.csv");
//
//        System.out.println("Excerpt of the dataframe content:");
//
//

        //
        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(
                        "userId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "foreName",
                        DataTypes.StringType,
                        false),
                DataTypes.createStructField(
                        "surName",
                        DataTypes.StringType,
                        false),
                DataTypes.createStructField(
                        "email",
                        DataTypes.StringType,
                        false),});

        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
//                .schema(schema)
                .option("inferSchema", "true")
                .load("src/main/resources/lab1/users.csv");
        //

        df.show(20);
        System.out.println("Dataframe's schema:");
        df.printSchema();
    }
}
