package org.lab7;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.unix_timestamp;

public class LogisticRegressionCppExam {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LogisticRegressionOnExam")
                .master("local[*]")
                .getOrCreate();

        String csvPath = "src/main/resources/lab7/egzamin-cpp.csv";

        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .option("inferSchema", "true")
                .csv(csvPath);

        Dataset<Row> typedDf = df
                .withColumn("OcenaC", col("OcenaC").cast(DataTypes.DoubleType))
                .withColumn("OcenaCpp", col("OcenaCpp").cast(DataTypes.DoubleType))
                .withColumn("Egzamin", col("Egzamin").cast(DataTypes.DoubleType));

        System.out.println("Schemat załadowanych danych:");
        typedDf.printSchema();

        Dataset<Row> processedDf = typedDf
                .withColumn("timestamp", unix_timestamp(col("DataC")))
                .withColumn("Wynik", expr("IF(Egzamin >= 3.0, 1, 0)").cast(DataTypes.IntegerType));

        System.out.println("Dane po wstępnym przetworzeniu:");
        processedDf.show(10, false);

        System.out.println("Schemat danych po przetworzeniu:");
        processedDf.printSchema();

        String[] featureCols = new String[]{"OcenaC", "timestamp", "OcenaCpp"};
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        Dataset<Row> assembledDf = assembler.transform(processedDf);

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.1)
                .setElasticNetParam(0)
                .setFeaturesCol("features")
                .setLabelCol("Wynik");

        LogisticRegressionModel lrModel = lr.fit(assembledDf);

        Vector coefficients = lrModel.coefficients();
        double intercept = lrModel.intercept();

        System.out.println("\n--- Równanie regresji logistycznej ---");
        String equation = String.format("logit(zdal) = %.6f*OcenaC + %.6f*timestamp + %.6f*OcenaCpp + %.6f",
                coefficients.apply(0),
                coefficients.apply(1),
                coefficients.apply(2),
                intercept);
        System.out.println(equation);

        System.out.println("\n--- Interpretacja współczynników ---");

        double coeffOcenaC = coefficients.apply(0);
        double oddsRatioOcenaC = Math.exp(coeffOcenaC);
        double percentageChangeOcenaC = (oddsRatioOcenaC - 1) * 100;
        System.out.printf("Wzrost OcenaC o 1 zwiększa logit o %.6f, a szanse zdania razy %.6f czyli o %.6f%%\n",
                coeffOcenaC, oddsRatioOcenaC, percentageChangeOcenaC);

        double coeffTimestamp = coefficients.apply(1);
        double logitChangePerDay = coeffTimestamp * 86400;
        double oddsRatioPerDay = Math.exp(logitChangePerDay);
        double percentageChangePerDay = (oddsRatioPerDay - 1) * 100;
        System.out.printf("Wzrost DataC o 1 dzień zwiększa logit o %.6f, a szanse zdania razy %.6f czyli o %.6f%%\n",
                logitChangePerDay, oddsRatioPerDay, percentageChangePerDay);

        double coeffOcenaCpp = coefficients.apply(2);
        double oddsRatioOcenaCpp = Math.exp(coeffOcenaCpp);
        double percentageChangeOcenaCpp = (oddsRatioOcenaCpp - 1) * 100;
        System.out.printf("Wzrost OcenaCpp o 1 zwiększa logit o %.6f, a szanse zdania razy %.6f czyli o %.6f%%\n",
                coeffOcenaCpp, oddsRatioOcenaCpp, percentageChangeOcenaCpp);

        spark.stop();
    }
}