package org.lab1;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import javax.print.DocFlavor;

import static org.apache.spark.sql.functions.*;

public class LoadMovies {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LoadMovies")
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
                .load("src/main/resources/lab1/movies.csv");
        // df2
 //
 //         var df2 = df
 //                 .withColumn("rok", year(now()))
 //                 .withColumn("miesiac", month(now()))
 //                 .withColumn("dzien", day(now()))
 //                 .withColumn("godzina", hour(now()));
 //
 //         df2.show(5);
 //         df2.printSchema();
 //
 //         df2.show(20);
 //         System.out.println("Dataframe's schema:");
 //         df2.printSchema();

        //

        var df_transformed = df
                .withColumn("title2", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 1)) // Extracts the title part
                .withColumn("year", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 2)) // Extracts the year part
                .withColumn("genres_array", split(col("genres"), "\\|")) // Extracts the year part
                .drop("genres")
                .drop("title") // Drops the original 'title' column
                .withColumnRenamed("title2", "title"); // Renames 'title2' to 'title'
//
//        df_transformed.show();
//        df_transformed.printSchema();

        var df_exploded = df
                .withColumn("genres_array", split(col("genres"), "\\|")) // Extracts the year part
                .drop("genres")
                .withColumn("genre", explode(col("genres_array"))) // Extracts the year part
                .drop("genres_array")
                .withColumn("title2", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 1)) // Extracts the title part
                .withColumn("year", regexp_extract(col("title"), "^(.*?)\\s*\\((\\d{4})\\)$", 2)) // Extracts the year part
                .drop("title") // Drops the original 'title' column
                .withColumnRenamed("title2", "title"); // Renames 'title2' to 'title'

//        df_exploded.show();
//        df_exploded.printSchema();
        
//        df_exploded.select("genre").distinct().show(false);
//
        var genreList = df_exploded.select("genre").distinct().as(Encoders.STRING()).collectAsList();
//        for(var s:genreList){
//            System.out.println(s);
//        }

        var df_multigenre = df_transformed;
        for(var s:genreList){
//            if(s.equals("(no genres listed)"))continue;
            df_multigenre=df_multigenre.withColumn(s,array_contains(col("genres_array"),s));
        }
        df_multigenre.show();
//

    }
}

