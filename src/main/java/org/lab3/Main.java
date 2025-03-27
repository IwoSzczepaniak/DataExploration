package org.lab3;

public class Main {
    public static void main(String[] args) {
        String[] resources = { "xy-001.csv", "xy-002.csv", "xy-003.csv", "xy-004.csv", "xy-005.csv", "xy-006.csv",
                "xy-007.csv", "xy-008.csv", "xy-009.csv", "xy-010.csv" };
        for (String resource : resources) {
            LinearRegressionPolynomialFeaturesOrderTwo.main(new String[] { resource });
        }
    }
}
