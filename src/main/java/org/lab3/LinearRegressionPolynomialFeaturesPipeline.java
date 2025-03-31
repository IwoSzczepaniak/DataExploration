package org.lab3;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.PolynomialExpansion;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.DataTypes;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.NumpyUtils;
import java.io.IOException;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

public class LinearRegressionPolynomialFeaturesPipeline {
    
    static void processDataset(SparkSession spark, String filename, int degree, Function<Double, Double> f_true) {
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/lab2/" + filename);

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"X"})
                .setOutputCol("features");

        PolynomialExpansion polyExpansion = new PolynomialExpansion()
                .setInputCol("features")
                .setOutputCol("polyFeatures")
                .setDegree(degree);

        LinearRegression lr = new LinearRegression()
                .setFeaturesCol("polyFeatures")
                .setLabelCol("Y")
                .setMaxIter(100)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {vectorAssembler, polyExpansion, lr});
        
        PipelineModel model = pipeline.fit(data);

        LinearRegressionModel lrModel = (LinearRegressionModel)model.stages()[2];
        
        System.out.printf("Coefficients for %s order %d: [%s]%n", 
            filename, 
            degree,
            Arrays.stream(lrModel.coefficients().toArray())
                  .mapToObj(coeff -> String.format("%.3f", coeff))
                  .collect(Collectors.joining(",")));
        System.out.println("Intercept: " + lrModel.intercept());
        System.out.println("R²: " + lrModel.summary().r2());
        System.out.println("MSE: " + lrModel.summary().meanSquaredError());

        List<Row> rows = data.collectAsList();
        List<Double> xValues = rows.stream()
                .map(row -> row.getDouble(0))
                .collect(Collectors.toList());
        List<Double> yValues = rows.stream()
                .map(row -> row.getDouble(1))
                .collect(Collectors.toList());

        plot(xValues, yValues, model, spark, "Polynomial Regression (degree " + degree + ") | " + filename, f_true);
    }

    static void plot(List<Double> x, List<Double> y, PipelineModel pipelineModel, SparkSession spark, 
            String title, Function<Double, Double> f_true) {
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        try {
            plt.plot().add(x, y, ".").label("data");

            double xmin = x.stream().mapToDouble(Double::doubleValue).min().getAsDouble();
            double xmax = x.stream().mapToDouble(Double::doubleValue).max().getAsDouble();
            double xdelta = 0.05 * (xmax - xmin);
            List<Double> fx = NumpyUtils.linspace(xmin - xdelta, xmax + xdelta, 100);

            List<Row> rows = fx.stream()
                    .map(x_val -> RowFactory.create(x_val))
                    .collect(Collectors.toList());

            StructType schema = new StructType().add("X", DataTypes.DoubleType);
            Dataset<Row> df_test = spark.createDataFrame(rows, schema);

            Dataset<Row> df_pred = pipelineModel.transform(df_test);
            List<Double> fy = df_pred.select("prediction")
                    .collectAsList()
                    .stream()
                    .map(row -> row.getDouble(0))
                    .collect(Collectors.toList());

            plt.plot().add(fx, fy).color("r").label("pred");

            if (f_true != null) {
                List<Double> fy_true = fx.stream()
                        .map(f_true::apply)
                        .collect(Collectors.toList());
                plt.plot().add(fx, fy_true).color("g").linestyle("--").label("$f_{true}$");
            }

            plt.title(title);
            plt.legend();
            plt.savefig("output/" + title.replaceAll("\\s+", "_") + ".png");
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        String[] resources = { "xy-001.csv", "xy-002.csv", "xy-003.csv", "xy-004.csv", "xy-005.csv", 
                             "xy-006.csv", "xy-007.csv", "xy-008.csv", "xy-009.csv", "xy-010.csv" };
        
        SparkSession spark = SparkSession.builder()
                .appName("Polynomial Regression Pipeline")
                .master("local[*]")
                .getOrCreate();

        try {
            for (String resource : resources) {
                System.out.println("Processing " + resource);
                processDataset(spark, resource, 3, null);
                System.out.println("--------------------------------");
            }
        } finally {
            spark.stop();
        }
    }
} 