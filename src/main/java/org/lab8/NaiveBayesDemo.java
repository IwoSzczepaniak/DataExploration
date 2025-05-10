package org.lab8;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.Vector;
import java.util.Locale;

import java.util.Arrays;
import java.util.List;

public class NaiveBayesDemo {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Naive Bayes Demo")
                .master("local[*]")
                .getOrCreate();

        StructType schema = new StructType()
                .add("author", DataTypes.StringType, false)
                .add("content", DataTypes.StringType, false);

        List<Row> rows = Arrays.asList(
                RowFactory.create("Ala", "aaa aaa bbb ccc"),
                RowFactory.create("Ala", "aaa bbb ddd"),
                RowFactory.create("Ala", "aaa bbb"),
                RowFactory.create("Ala", "aaa bbb bbb"),
                RowFactory.create("Ola", "aaa ccc ddd"),
                RowFactory.create("Ola", "bbb ccc ddd"),
                RowFactory.create("Ola", "ccc ddd eee")
        );

        Dataset<Row> df = spark.createDataFrame(rows, schema);

        String sep = "[\\s\\p{Punct}\\u2014\\u2026\\u201C\\u201E]+";
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("content")
                .setOutputCol("words")
                .setPattern(sep);
        df = tokenizer.transform(df);
        
        System.out.println("After tokenization:");
        df.show();
        System.out.println("-----------");

        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("features")
                .setVocabSize(10_000)
                .setMinDF(1);

        CountVectorizerModel countVectorizerModel = countVectorizer.fit(df);
        df = countVectorizerModel.transform(df);

        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("author")
                .setOutputCol("label");

        StringIndexerModel labelModel = labelIndexer.fit(df);
        df = labelModel.transform(df);
        
        System.out.println("After feature extraction and label indexing:");
        df.show();

        NaiveBayes nb = new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setModelType("multinomial")
                .setSmoothing(0.01);

        System.out.println("Naive Bayes parameters explanation:");
        System.out.println(nb.explainParams());

        NaiveBayesModel model = nb.fit(df);

        System.out.println("\nModel Parameters Analysis");
        System.out.println("Vocabulary and Labels:");
        String[] vocab = countVectorizerModel.vocabulary();
        String[] labels = labelModel.labels();
        
        System.out.println("\nVocabulary:");
        for (int i = 0; i < vocab.length; i++) {
            System.out.printf("%d: %s%n", i, vocab[i]);
        }
        
        System.out.println("\nLabels:");
        for (int i = 0; i < labels.length; i++) {
            System.out.printf("%d: %s%n", i, labels[i]);
        }

        System.out.println("\nConditional Probabilities (Likelihood):");
        Matrix theta = model.theta();
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < vocab.length; j++) {
                double prob = Math.exp(theta.apply(i, j));
                System.out.printf("P(%s|%s)=%.6f (log=%.6f)%n", 
                    vocab[j], labels[i], prob, theta.apply(i, j));
            }
        }

        System.out.println("\nPrior Probabilities:");
        Vector pi = model.pi();
        for (int i = 0; i < labels.length; i++) {
            double prob = Math.exp(pi.apply(i));
            System.out.printf("P(%s)=%.6f (log=%.6f)%n", 
                labels[i], prob, pi.apply(i));
        }

        System.out.println("\nPrediction Test with smoothing=0.01");
        DenseVector testData = new DenseVector(new double[]{1,0,2,1,0});
        
        double p0 = pi.apply(0);
        double p1 = pi.apply(1);
        
        for (int j = 0; j < testData.size(); j++) {
            if (testData.apply(j) != 0) {
                p0 += testData.apply(j) * theta.apply(0, j);
                p1 += testData.apply(j) * theta.apply(1, j);
            }
        }

        System.out.printf(Locale.US, "Manual calculation:\nlog(p0)=%g p0=%g\nlog(p1)=%g p1=%g\n",
                p0, Math.exp(p0),
                p1, Math.exp(p1));
        System.out.println("Manual classification result: " + (p0 > p1 ? 0 : 1));

        Vector proba = model.predictRaw(testData);
        System.out.println("\nModel raw probabilities:");
        System.out.printf("Pr:[%.6f, %.6f]%n", 
            Math.exp(proba.apply(0)), Math.exp(proba.apply(1)));
        
        double predLabel = model.predict(testData);
        System.out.printf("Model predicted Label: %.0f (%s)%n", 
            predLabel, labels[(int)predLabel]);

        System.out.println("\nTesting with smoothing = 0:");
        NaiveBayes nbNoSmooth = new NaiveBayes()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setModelType("multinomial")
                .setSmoothing(0.0);

        NaiveBayesModel modelNoSmooth = nbNoSmooth.fit(df);
        
        System.out.println("\nConditional Probabilities with no smoothing:");
        Matrix thetaNoSmooth = modelNoSmooth.theta();
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < vocab.length; j++) {
                double prob = Math.exp(thetaNoSmooth.apply(i, j));
                System.out.printf("P(%s|%s)=%.6f (log=%.6f)%n", 
                    vocab[j], labels[i], prob, thetaNoSmooth.apply(i, j));
            }
        }

        Vector probaNoSmooth = modelNoSmooth.predictRaw(testData);
        System.out.println("\nRaw probabilities with no smoothing:");
        System.out.printf("Pr:[%.6f, %.6f]%n", 
            Math.exp(probaNoSmooth.apply(0)), Math.exp(probaNoSmooth.apply(1)));

        spark.stop();
    }
}