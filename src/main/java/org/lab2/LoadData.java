package org.lab2;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import com.github.sh0nk.matplotlib4j.NumpyUtils;

public class LoadData {
    static void plotObjectiveHistory(double[] lossHistory, String params) {
        List<Double> x = IntStream.range(0, lossHistory.length).mapToDouble(d -> d).boxed().toList();
        List<Double> lossList = Arrays.stream(lossHistory).boxed().toList();

        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        try {
            plt.plot().add(x, lossList).label("loss");
            plt.xlabel("Iteration");
            plt.ylabel("Loss");
            if (params != null) {
                plt.title("Loss history " + params);
            } else {
                plt.title("Loss history");
            }
            plt.legend();
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     *
     * @param x       - współrzedne x danych
     * @param y       - współrzedne y danych
     * @param lrModel - model regresji
     * @param title   - tytuł do wyswietlenia (może być null)
     * @param f_true  - funkcja f_true (może być null)
     */
    static void plot(List<Double> x, List<Double> y, LinearRegressionModel lrModel, String title,
            Function<Double, Double> f_true) {
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        try {
            plt.plot().add(x, y, ".").label("data");

            double xmin = x.stream().mapToDouble(Double::doubleValue).min().getAsDouble();
            double xmax = x.stream().mapToDouble(Double::doubleValue).max().getAsDouble();
            double xdelta = 0.05 * (xmax - xmin);

            List<Double> fx = NumpyUtils.linspace(xmin - xdelta, xmax + xdelta, 100);

            List<Double> fy = fx.stream()
                    .map(x_val -> {
                        double[] arr = new double[] { x_val };
                        return lrModel.predict(new DenseVector(arr));
                    })
                    .collect(Collectors.toList());

            plt.plot().add(fx, fy).color("r").label("pred");

            if (f_true != null) {
                List<Double> fy_true = fx.stream()
                        .map(f_true::apply)
                        .collect(Collectors.toList());
                plt.plot().add(fx, fy_true).color("g").linestyle("--").label("$f_{true}$");
            }

            if (title != null) {
                plt.title(title);
            } else {
                plt.title("Linear regression");
            }

            plt.legend();
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    static void trainAndEvaluateModel(Dataset<Row> vectorData, double regParam, double elasticNetParam,
            List<Double> xValues, List<Double> yValues, Function<Double, Double> f_true) {

        String params = String.format(" regParam=%.1f, elasticNetParam=%.1f", regParam, elasticNetParam);

        // 1.2
        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFeaturesCol("features")
                .setLabelCol("Y");

        LinearRegressionModel lrModel = lr.fit(vectorData);

        System.out.println("Coefficients: " + lrModel.coefficients());
        System.out.println("Intercept: " + lrModel.intercept());

        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show(100);
        System.out.println("MSE: " + trainingSummary.meanSquaredError());
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("MAE: " + trainingSummary.meanAbsoluteError());
        System.out.println("r2: " + trainingSummary.r2());

        plotObjectiveHistory(trainingSummary.objectiveHistory(), "" + params);

        // // 1.3
        // plot(xValues, yValues, lrModel, "Linear regression", null);

        // 1.3 extra
        plot(xValues, yValues, lrModel, "Linear regression" + params, f_true);
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Load XY Data")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/lab2/xy-001.csv");

        System.out.println("Oryginalne dane:");
        data.show(5);
        data.printSchema();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] { "X" })
                .setOutputCol("features");

        Dataset<Row> vectorData = vectorAssembler.transform(data);

        System.out.println("\nDane po transformacji:");
        vectorData.show(5);
        vectorData.printSchema();

        List<Row> rows = data.collectAsList();
        List<Double> xValues = rows.stream()
                .map(row -> row.getDouble(0))
                .collect(Collectors.toList());
        List<Double> yValues = rows.stream()
                .map(row -> row.getDouble(1))
                .collect(Collectors.toList());

        // 1.4
        double[] regParams = { 0.0, 10.0, 20.0, 50.0, 100.0 };
        double elasticNetParam = 0.8;
        Function<Double, Double> f_true = x -> 2.5 * x + 1;

        for (double regParam : regParams) {
            trainAndEvaluateModel(vectorData, regParam, elasticNetParam, xValues, yValues, f_true);
        }

        spark.stop();
    }
}
