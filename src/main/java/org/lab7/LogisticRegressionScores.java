package org.lab7;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LogisticRegressionScores {
    static void plotObjectiveHistory(double[] objectiveHistory) {
        System.out.println("\n--- Objective History ---");
        if (objectiveHistory == null || objectiveHistory.length == 0) {
            System.out.println("No objective history available.");
            return;
        }

        List<Integer> iterations = IntStream.range(0, objectiveHistory.length)
                                            .boxed().collect(Collectors.toList());
        List<Double> objectives = new ArrayList<>();
        for (double objective : objectiveHistory) {
            objectives.add(objective);
        }

        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.plot()
           .add(iterations, objectives)
           .linestyle("-");
        plt.xlabel("Iteration");
        plt.ylabel("Objective Value");
        plt.title("Objective Function History");
        plt.legend();

        try {
            plt.show();
            System.out.println("Objective history chart displayed using matplotlib4j.");
        } catch (IOException | PythonExecutionException e) {
            System.err.println("Error displaying objective history chart with matplotlib4j: " + e.getMessage());
            System.err.println("Please ensure Python, Matplotlib are installed and python3 command is correct.");
            System.out.println("Fallback: Printing objective history values:");
            System.out.println("Iteration | Objective Value");
            System.out.println("-------------------------");
            for (int i = 0; i < objectiveHistory.length; i++) {
                System.out.printf("%-9d | %.6f%n", i, objectiveHistory[i]);
            }
        }
    }

    static void plotROC(Dataset<Row> roc) {
        System.out.println("\n--- ROC Curve Data (for plotting) ---");
        List<Row> rocData = roc.collectAsList();
        if (rocData == null || rocData.isEmpty()) {
            System.out.println("No ROC data available to plot.");
            return;
        }

        List<Double> fprList = new ArrayList<>();
        List<Double> tprList = new ArrayList<>();
        System.out.println("FPR         | TPR");
        System.out.println("--------------------");
        for (Row r : rocData) {
            Object fprObj = r.get(0);
            Object tprObj = r.get(1);

            double fpr = (fprObj instanceof Number) ? ((Number) fprObj).doubleValue() : 0.0;
            double tpr = (tprObj instanceof Number) ? ((Number) tprObj).doubleValue() : 0.0;
            
            fprList.add(fpr);
            tprList.add(tpr);
            System.out.printf("%.10f | %.10f%n", fpr, tpr);
        }

        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));
        plt.plot()
           .add(fprList, tprList)
           .linestyle("-")
           .label("ROC Curve");
        plt.xlabel("False Positive Rate (FPR)");
        plt.ylabel("True Positive Rate (TPR)");
        plt.title("ROC Curve");
        plt.legend();

        try {
            plt.show();
            System.out.println("ROC chart displayed using matplotlib4j.");
        } catch (IOException | PythonExecutionException e) {
            System.err.println("Error displaying ROC chart with matplotlib4j: " + e.getMessage());
            System.err.println("Please ensure Python, Matplotlib are installed and python3 command is correct.");
        }
    }

    public static LogisticRegressionModel trainAndTest(Dataset<Row> df) {
        System.out.println("\n--- Starting Train/Test and Evaluation ---");
        
        int splitSeed = 123;
        Dataset<Row>[] splits = df.randomSplit(new double[]{0.7, 0.3}, splitSeed);
        Dataset<Row> df_train = splits[0];
        Dataset<Row> df_test = splits[1];

        System.out.println("Training set size: " + df_train.count());
        System.out.println("Test set size: " + df_test.count());

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(20)
                .setRegParam(0.1)
                .setFeaturesCol("features")
                .setLabelCol("Wynik");

        LogisticRegressionModel lrModel = lr.fit(df_train);
        System.out.println("Model trained on the training set.");

        System.out.println("\n--- Training Summary Analysis ---");
        BinaryLogisticRegressionTrainingSummary trainingSummary = lrModel.binarySummary();

        double[] objectiveHistory = trainingSummary.objectiveHistory();
        plotObjectiveHistory(objectiveHistory);

        Dataset<Row> roc = trainingSummary.roc();
        System.out.println("\n--- ROC Curve Table (Training Set) ---");
        roc.show();

        plotROC(roc);

        System.out.println("\n--- Performance Metrics (Training Set) ---");
        System.out.printf("Accuracy: %.4f%n", trainingSummary.accuracy());
        System.out.printf("Area Under ROC (AUC): %.4f%n", trainingSummary.areaUnderROC());
        System.out.printf("Weighted FPR: %.4f%n", trainingSummary.weightedFalsePositiveRate());
        System.out.printf("Weighted TPR (Recall): %.4f%n", trainingSummary.weightedTruePositiveRate());
        System.out.printf("Weighted Precision: %.4f%n", trainingSummary.weightedPrecision());
        System.out.printf("Weighted Recall: %.4f%n", trainingSummary.weightedRecall());
        System.out.printf("Weighted F-measure (beta=1.0): %.4f%n", trainingSummary.weightedFMeasure());

        System.out.println("\n--- Threshold Selection based on F-Measure (Training Set) ---");
        Dataset<Row> df_fmeasures = trainingSummary.fMeasureByThreshold();

        double maxFMeasure = df_fmeasures.select(functions.max("F-Measure")).head().getDouble(0);

        double bestThreshold = df_fmeasures.where(functions.col("F-Measure").equalTo(maxFMeasure))
                .select("threshold").head().getDouble(0);

        System.out.printf("Max F-Measure on Training Set: %.4f%n", maxFMeasure);
        System.out.printf("Corresponding Threshold: %.6f%n", bestThreshold);

        lrModel.setThreshold(bestThreshold);
        System.out.printf("Set model threshold to: %.6f%n", lrModel.getThreshold());

        System.out.println("\n--- Evaluating on Test Set (with updated threshold) ---");

        Dataset<Row> predictions_test = lrModel.transform(df_test);
        
        MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator()
                .setLabelCol("Wynik")
                .setPredictionCol("prediction");

        double accuracy = eval.setMetricName("accuracy").evaluate(predictions_test);
        double weightedPrecision = eval.setMetricName("weightedPrecision").evaluate(predictions_test);
        double weightedRecall = eval.setMetricName("weightedRecall").evaluate(predictions_test);
        double f1 = eval.setMetricName("f1").evaluate(predictions_test);

        System.out.println("\n--- Performance Metrics (Test Set) ---");
        System.out.printf("Accuracy:          %.4f%n", accuracy);
        System.out.printf("Weighted Precision: %.4f%n", weightedPrecision);
        System.out.printf("Weighted Recall:    %.4f%n", weightedRecall);
        System.out.printf("F1 Score:          %.4f%n", f1);

        System.out.println("\n--- Finished Train/Test and Evaluation ---");
        return lrModel;
    }
    
} 