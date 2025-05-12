package org.lab7;

import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.*;

public class LogisticRegressionGrid {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("LogisticRegressionGrid")
                .master("local[*]")
                .getOrCreate();

        String examCsvPath = "src/main/resources/lab7/egzamin-cpp.csv";
        Dataset<Row> dfExam = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .option("inferSchema", "true")
                .csv(examCsvPath);

        Dataset<Row> typedDfExam = dfExam
                .withColumn("OcenaC", col("OcenaC").cast(DataTypes.DoubleType))
                .withColumn("OcenaCpp", col("OcenaCpp").cast(DataTypes.DoubleType))
                .withColumn("Egzamin", col("Egzamin").cast(DataTypes.DoubleType));

        Dataset<Row> processedDfExam = typedDfExam
                .withColumn("timestamp", unix_timestamp(col("DataC")))
                .withColumn("Wynik", expr("IF(Egzamin >= 3.0, 1, 0)").cast(DataTypes.IntegerType));

        String[] featureCols = new String[]{"OcenaC", "timestamp", "OcenaCpp"};
        VectorAssembler assemblerExam = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        Dataset<Row> assembledDfExam = assemblerExam.transform(processedDfExam);
        System.out.println("Exam data preprocessed and features assembled.");

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.1)
                .setFeaturesCol("features")
                .setLabelCol("Wynik");

        LogisticRegressionModel lrModelGrid = lr.fit(assembledDfExam);
        System.out.println("Model trained on full exam dataset.");

        BinaryLogisticRegressionTrainingSummary trainingSummary = lrModelGrid.binarySummary();
        Dataset<Row> df_fmeasures = trainingSummary.fMeasureByThreshold();
        double maxFMeasure = df_fmeasures.select(functions.max("F-Measure")).head().getDouble(0);
        double bestThreshold = df_fmeasures.where(functions.col("F-Measure").equalTo(maxFMeasure))
                .select("threshold").head().getDouble(0);
        lrModelGrid.setThreshold(bestThreshold);
        System.out.printf("Optimal threshold based on F-Measure (%.4f) found: %.6f. Set on model.%n", maxFMeasure, bestThreshold);

        addClassificationToGrid(spark, lrModelGrid);

        spark.stop();
    }

    static void addClassificationToGrid(SparkSession spark, LogisticRegressionModel lrModel) {
        System.out.println("\n--- Starting Grid Classification --- ");

        String gridCsvPath = "src/main/resources/lab7/grid.csv";
        Dataset<Row> dfGrid = spark.read()
                .option("header", "true")
                .option("delimiter", ",")
                .option("inferSchema", "true")
                .csv(gridCsvPath);
        System.out.println("Loaded grid data:");
        dfGrid.show(5);
        System.out.println("Schema inferred from grid.csv:");
        dfGrid.printSchema();

        Dataset<Row> typedDfGrid = dfGrid
                .withColumn("OcenaC", col("OcenaC").cast(DataTypes.DoubleType))
                .withColumn("OcenaCpp", col("OcenaCpp").cast(DataTypes.DoubleType));

        Dataset<Row> processedDfGrid = typedDfGrid
                .withColumn("timestamp", unix_timestamp(col("DataC")));

        String[] featureCols = new String[]{"OcenaC", "timestamp", "OcenaCpp"};
        VectorAssembler assemblerGrid = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        Dataset<Row> assembledDfGrid = assemblerGrid.transform(processedDfGrid);
        System.out.println("Grid data preprocessed and features assembled.");

        Dataset<Row> predictionsGrid = lrModel.transform(assembledDfGrid);
        System.out.println("Predictions generated for grid data.");

        Dataset<Row> dfFinalGrid = predictionsGrid
                .withColumn("Wynik", expr("IF(prediction = 1.0, 'Zdał', 'Nie zdał')"))
                .select("ImieNazwisko", "OcenaC", "DataC", "OcenaCpp", "Wynik");

        System.out.println("\n--- Final Grid with Classification ---");
        dfFinalGrid.show(20, false);

        System.out.println("\nSaving grid results to CSV...");
        dfFinalGrid.repartition(1)
                   .write()
                   .format("csv")
                   .option("header", true)
                   .option("delimiter", ";")
                   .mode(SaveMode.Overwrite)
                   .save("output/grid-with-classification.csv");
        System.out.println("Grid results saved to output/grid-with-classification.csv");
    }
}