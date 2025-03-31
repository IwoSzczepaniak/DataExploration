package org.lab3;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
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
import org.apache.spark.sql.Encoders;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.NumpyUtils;
import java.io.IOException;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

public class LinearRegressionPolynomialFeaturesPipelineTrainTestSplit {
    
    static void processDataset(SparkSession spark, String filename, int degree, Function<Double, Double> f_true, 
            boolean useRandomSplit, boolean shuffleBeforeSplit) {
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("src/main/resources/lab2/" + filename);

        if (shuffleBeforeSplit) {
            data = data.orderBy(org.apache.spark.sql.functions.rand(3));
        }

        Dataset<Row> df_train, df_test;
        if (useRandomSplit) {
            Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
            df_train = splits[0];
            df_test = splits[1];
        } else {
            long rowsCount = data.count();
            int trainCount = (int)(rowsCount * 0.7);
            df_train = data.select("*").limit(trainCount);
            df_test = data.select("*").offset(trainCount);
        }

        System.out.println("Training set size: " + df_train.count());
        System.out.println("Test set size: " + df_test.count());

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
        
        PipelineModel model = pipeline.fit(df_train);

        LinearRegressionModel lrModel = (LinearRegressionModel)model.stages()[2];
        
        System.out.printf("Coefficients for %s order %d: [%s]%n", 
            filename, 
            degree,
            Arrays.stream(lrModel.coefficients().toArray())
                  .mapToObj(coeff -> String.format("%.3f", coeff))
                  .collect(Collectors.joining(",")));
        System.out.println("Intercept: " + lrModel.intercept());
        System.out.println("Training R²: " + lrModel.summary().r2());
        System.out.println("Training MSE: " + lrModel.summary().meanSquaredError());

        List<Double> x_train = df_train.select("X").as(Encoders.DOUBLE()).collectAsList();
        List<Double> y_train = df_train.select("Y").as(Encoders.DOUBLE()).collectAsList();
        String splitType = useRandomSplit ? "random" : (shuffleBeforeSplit ? "shuffled" : "fixed");
        plot(x_train, y_train, model, spark, 
            String.format("Linear regression: %s (%s split, training data)", filename, splitType), f_true);

        List<Double> x_test = df_test.select("X").as(Encoders.DOUBLE()).collectAsList();
        List<Double> y_test = df_test.select("Y").as(Encoders.DOUBLE()).collectAsList();
        plot(x_test, y_test, model, spark, 
            String.format("Linear regression: %s (%s split, test data)", filename, splitType), f_true);

        Dataset<Row> df_test_prediction = model.transform(df_test);
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Y")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(df_test_prediction);
        evaluator.setMetricName("r2");
        double r2 = evaluator.evaluate(df_test_prediction);
        
        System.out.println("Test RMSE: " + rmse);
        System.out.println("Test R²: " + r2);
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
        String[] resources = { "xy-003.csv", "xy-005.csv" };
        int[] degrees = { 3, 2 }; // degree 3 for xy-003.csv, degree 2 for xy-005.csv
        
        SparkSession spark = SparkSession.builder()
                .appName("Polynomial Regression Pipeline with Train-Test Split")
                .master("local[*]")
                .getOrCreate();

        try {
            for (int i = 0; i < resources.length; i++) {
                String resource = resources[i];
                int degree = degrees[i];
                System.out.println("\nProcessing " + resource + " with degree " + degree);
                
                System.out.println("\nFixed split:");
                processDataset(spark, resource, degree, null, false, false);
                
                System.out.println("\nRandom split:");
                processDataset(spark, resource, degree, null, true, false);
                
                System.out.println("\nShuffled fixed split:");
                processDataset(spark, resource, degree, null, false, true);
                
                System.out.println("--------------------------------");
            }
        } finally {
            spark.stop();
        }
    }
}