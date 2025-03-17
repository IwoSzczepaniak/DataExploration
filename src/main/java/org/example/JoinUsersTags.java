package org.example;

import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;
import java.util.List;

public class JoinUsersTags {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("JoinUsersTags")
                .master("local")
                .getOrCreate();
        System.out.println("Using Apache Spark v" + spark.version());

        Dataset<Row> df_users = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/users.csv");

        Dataset<Row> df_tags = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("src/main/resources/tags.csv");

        df_users.createOrReplaceTempView("users");
        df_tags.createOrReplaceTempView("tags");

        String query = """
                SELECT u.email, t.tag
                FROM users u
                JOIN tags t ON u.userId = t.userId
                """;

        Dataset<Row> df_ut = spark.sql(query);

        Dataset<Row> df_grouped = df_ut.groupBy("email")
                .agg(concat_ws(" ", collect_list("tag")).alias("tags"));

        df_grouped.show();

        List<Row> tagsList = df_grouped.select("tags").collectAsList();
        for (Row row : tagsList) {
            System.out.println(row.getString(0));
        }
    }
}