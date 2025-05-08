package org.lab8;

class EvaluationResult {
    String filename;
    double accuracy;
    double weightedPrecision;
    double weightedRecall;
    double f1;

    EvaluationResult(String filename, double accuracy, double weightedPrecision, double weightedRecall, double f1) {
        this.filename = filename;
        this.accuracy = accuracy;
        this.weightedPrecision = weightedPrecision;
        this.weightedRecall = weightedRecall;
        this.f1 = f1;
    }

    public void printResult() {
            String shortFileName = filename.substring(filename.lastIndexOf("/") + 1);
            System.out.printf("%-45s | %-10.4f | %-10.4f | %-10.4f | %-10.4f%n",
                    shortFileName,
                    accuracy,
                    weightedPrecision,
                    weightedRecall,
                    f1);
    }
}