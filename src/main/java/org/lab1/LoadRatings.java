package org.lab1;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;

import static org.apache.spark.sql.functions.*;

public class LoadRatings {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadRatings")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());


        //
        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField(
                        "userId",
                        DataTypes.IntegerType,
                        true),
                DataTypes.createStructField(
                        "movieId",
                        DataTypes.StringType,
                        false),
                DataTypes.createStructField(
                        "rating",
                        DataTypes.StringType,
                        false),
                DataTypes.createStructField(
                        "timestamp",
                        DataTypes.StringType,
                        false),});

        Dataset<Row> df = spark.read()
                .format("csv")
                .option("header", "true")
//                .schema(schema)
                .option("inferSchema", "true")
                .load("src/main/resources/lab1/ratings.csv");
        //

        var df2 = df.withColumn("datetime", functions.from_unixtime(df.col("timestamp")))
                .withColumn("year", functions.year(col("datetime")))
                .withColumn("month", functions.month(col("datetime")))
                .withColumn("day", functions.dayofmonth(col("datetime")));

        var df_stats_ym = df2.groupBy("year", "month").count().orderBy("year", "month");

        df_stats_ym.show(20);
        System.out.println("Dataframe's schema:");
        df_stats_ym.printSchema();

        Helper.plot_stats_ym(df_stats_ym, "Ratings", "Count");

    }
}
