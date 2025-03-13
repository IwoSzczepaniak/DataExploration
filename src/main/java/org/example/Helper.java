package org.example;

import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonConfig;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;

import java.io.IOException;

import static org.apache.spark.sql.functions.*;

public class Helper {
    static void plot_stats_ym(Dataset<Row> df, String title, String label) {
        var labels = df.select(concat(col("year"), lit("-"), col("month"))).as(Encoders.STRING()).collectAsList();
        var x = NumpyUtils.arange(0, labels.size() - 1, 1);
        x = df.select(expr("year+(month-1)/12")).as(Encoders.DOUBLE()).collectAsList();
        var y = df.select("count").as(Encoders.DOUBLE()).collectAsList();


        // Plot plt = Plot.create();
        // Pod linuxem może być potrzebne
        Plot plt = Plot.create(PythonConfig.pythonBinPathConfig("python3"));


        plt.plot().add(x, y).linestyle("-").label(label);
        plt.legend();
        plt.title(title);
        try {
            plt.show();
        } catch (IOException | PythonExecutionException e) {
            throw new RuntimeException(e);
        }
    }
}
