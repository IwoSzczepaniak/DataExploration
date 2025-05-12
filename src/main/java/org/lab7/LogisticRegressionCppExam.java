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

        Dataset<Row> df_with_predictions = lrModel.transform(assembledDf);
        Dataset<Row> df_predictions = df_with_predictions
                .select("features", "rawPrediction", "probability", "prediction", "Wynik");

        System.out.println("\n--- Predykcje ---");
        df_predictions.show(10, false);

        analyzePredictions(df_predictions, lrModel);

        spark.stop();
    }

    private static void analyzePredictions(Dataset<Row> dfPredictions, LogisticRegressionModel lrModel) {
        org.apache.spark.ml.linalg.DenseVector coefficientsDense = (org.apache.spark.ml.linalg.DenseVector) lrModel.coefficients();
        double intercept = lrModel.intercept();

        dfPredictions.foreach(row -> {
            Vector features = row.getAs("features");
            Vector rawPrediction = row.getAs("rawPrediction");
            Vector probability = row.getAs("probability");
            double prediction = row.getAs("prediction");
            int actualWynik = row.getAs("Wynik");

            double logit = 0.0;
            for (int i = 0; i < features.size(); i++) {
                logit += coefficientsDense.apply(i) * features.apply(i);
            }
            logit += intercept;

            double prob1_calculated = 1.0 / (1.0 + Math.exp(-logit));
            double prob0_calculated = Math.exp(-logit) / (1.0 + Math.exp(-logit));


            System.out.println("\n--- Analiza predykcji dla wiersza ---");
            System.out.println("Cechy: " + features.toString());
            System.out.println("Rzeczywisty wynik: " + actualWynik);
            System.out.println("Predykcja modelu: " + prediction);

            System.out.printf("Obliczony logit: %.6f\n", logit);
            System.out.printf("rawPrediction (Spark): [%.6f, %.6f]\n", rawPrediction.apply(0), rawPrediction.apply(1));

            System.out.printf("Obliczone P(Y=1): %.6f, P(Y=0): %.6f\n", prob1_calculated, prob0_calculated);
            System.out.printf("Prawdopodobieństwa (Spark): P(Y=0)=%.6f (indeks 0), P(Y=1)=%.6f (indeks 1)\n", probability.apply(0), probability.apply(1));

            double predictedLabelProbability;
            if (prediction == 1.0) {
                predictedLabelProbability = probability.apply(1); // P(Y=1)
            } else {
                predictedLabelProbability = probability.apply(0); // P(Y=0)
            }
            System.out.printf("Prawdopodobieństwo dla predykcji (%.0f): %.6f\n", prediction, predictedLabelProbability);
        });
    }
}