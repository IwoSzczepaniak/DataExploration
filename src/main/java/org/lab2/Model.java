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

public class Model {
    static void plotObjectiveHistory(double[] lossHistory, String params) {
        List<Double> x = IntStream.range(0, lossHistory.length).mapToDouble(d -> d).boxed().toList();
        List<Double> lossList = Arrays.stream(lossHistory).boxed().toList();

        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        try {
            plt.plot().add(x, lossList).label("loss");
            plt.xlabel("Iteration");
            plt.ylabel("Loss");
            String title = params != null ? "Loss history " + params : "Loss history";
            plt.title(title);
            plt.legend();
            plt.savefig("output/" + title.replaceAll("\\s+", "_") + ".png");
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
            Function<Double, Double> f_true, int order) {
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        try {
            plt.plot().add(x, y, ".").label("data");

            double xmin = x.stream().mapToDouble(Double::doubleValue).min().getAsDouble();
            double xmax = x.stream().mapToDouble(Double::doubleValue).max().getAsDouble();
            double xdelta = 0.05 * (xmax - xmin);

            List<Double> fx = NumpyUtils.linspace(xmin - xdelta, xmax + xdelta, 100);

            List<Double> fy = fx.stream()
                    .map(x_val -> {
                        double[] features = new double[order];
                        for (int i = 0; i < order; i++) {
                            features[i] = Math.pow(x_val, i + 1);
                        }
                        return lrModel.predict(new DenseVector(features));
                    })
                    .collect(Collectors.toList());

            plt.plot().add(fx, fy).color("r").label("pred");

            if (f_true != null) {
                List<Double> fy_true = fx.stream()
                        .map(f_true::apply)
                        .collect(Collectors.toList());
                plt.plot().add(fx, fy_true).color("g").linestyle("--").label("$f_{true}$");
            }

            String plotTitle = title != null ? title : "Polynomial regression (order " + order + ")";
            plt.title(plotTitle);
            plt.legend();
            plt.savefig("output/" + plotTitle.replaceAll("\\s+", "_") + ".png");
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    public static void trainAndEvaluate(Dataset<Row> vectorData, double regParam, double elasticNetParam,
            int maxIter, List<Double> xValues, List<Double> yValues, Function<Double, Double> f_true, String resourceName, int order) {

        String params = String.format(" regP=%.1f, elasticNetP=%.1f, maxIter=%d", regParam, elasticNetParam, maxIter);

        // 1.2
        LinearRegression lr = new LinearRegression()
                .setMaxIter(maxIter)
                .setRegParam(regParam)
                .setElasticNetParam(elasticNetParam)
                .setFeaturesCol("features")
                .setLabelCol("Y");

        LinearRegressionModel lrModel = lr.fit(vectorData);

        System.out.println("Coefficients: " + lrModel.coefficients());
        System.out.println("Intercept: " + lrModel.intercept());

        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show(5);
        System.out.println("MSE: " + trainingSummary.meanSquaredError());
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("MAE: " + trainingSummary.meanAbsoluteError());
        System.out.println("r2: " + trainingSummary.r2());

        plotObjectiveHistory(trainingSummary.objectiveHistory(), params + " | " + resourceName);

        // // 1.3
        // plot(xValues, yValues, lrModel, "Linear regression", null);

        // 1.3 extra
        plot(xValues, yValues, lrModel, "Pol reg (order: " + order + ")" + params + " | " + resourceName, f_true, order);
    }

    public static void main(String[] args) {
        var resourceName = args[0];
        System.out.println("Resource name: " + resourceName);
        SparkSession spark = SparkSession.builder()
                .appName("Load XY Data")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/lab2/" + resourceName);

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
        // double[] regParams = { 0.0, 10.0, 20.0, 50.0, 100.0 };
        // double elasticNetParam = 0.8;
        // Function<Double, Double> f_true = x -> 2.5 * x + 1;

        // for (double regParam : regParams) {
        //     trainAndEvaluateModel(vectorData, regParam, elasticNetParam, xValues, yValues, null, resourceName);
        // }

        // 3
        trainAndEvaluate(vectorData, 10.0, 0.8, 100, xValues, yValues, null, resourceName, 1);

        spark.stop();
    }
}
