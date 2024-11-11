import infodynamics.utils.ArrayFileReader;
import infodynamics.utils.AnalyticMeasurementDistribution;
import infodynamics.utils.MatrixUtils;
import infodynamics.measures.discrete.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.*;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class MutualInfo extends JFrame {

    private static final String[] variableNames = {
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
    };

    static class MutualInfoResult {
        String source;
        String destination;
        double mutualInfoValue;

        public MutualInfoResult(String source, String destination, double mutualInfoValue) {
            this.source = source;
            this.destination = destination;
            this.mutualInfoValue = mutualInfoValue;
        }
    }

    public MutualInfo(String title, double[] mutualInfoResults, String sourceColumn) {
        super(title);

        DefaultCategoryDataset dataset = createDataset(mutualInfoResults, sourceColumn);

        JFreeChart chart = ChartFactory.createBarChart(
                "Mutual Information Results (Source: " + sourceColumn + ")",
                "Destination|Source",
                "Mutual Information (bits)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 600));
        setContentPane(chartPanel);
    }

    private DefaultCategoryDataset createDataset(double[] mutualInfoResults, String sourceColumn) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < mutualInfoResults.length; i++) {
            if (mutualInfoResults[i] != -1) {
                String label = variableNames[i] + "|" + sourceColumn;
                dataset.addValue(mutualInfoResults[i], "Mutual Information", label);
            }
        }
        return dataset;
    }

    public static void main(String[] args) throws Exception {
        String dataFile = "C:\\Users\\Eric\\Desktop\\iris\\encoded_adult_classification.data";
        ArrayFileReader afr = new ArrayFileReader(dataFile);
        int[][] data = afr.getInt2DMatrix();
        int[] Cvariables = {4, 9, 4, 16, 4, 7, 15, 6, 5, 2, 2, 5, 4, 42, 2};

        List<MutualInfoResult> mutualInfoResultsList = new ArrayList<>();

        for (int sourceIndex = 0; sourceIndex < data[0].length; sourceIndex++) {
            int[] source = MatrixUtils.selectColumn(data, sourceIndex);

            double[] mutualInfoResults = new double[data[0].length];

            for (int i = 0; i < mutualInfoResults.length; i++) {
                mutualInfoResults[i] = -1;
            }

            for (int destIndex = sourceIndex + 1; destIndex < data[0].length; destIndex++) {
                int[] destination = MatrixUtils.selectColumn(data, destIndex);

                MutualInformationCalculatorDiscrete calc = new MutualInformationCalculatorDiscrete(Cvariables[sourceIndex], Cvariables[destIndex], 0);
                calc.initialise();
                calc.addObservations(source, destination);
                double result = calc.computeAverageLocalOfObservations();
                mutualInfoResults[destIndex] = result;
                mutualInfoResultsList.add(new MutualInfoResult(variableNames[sourceIndex], variableNames[destIndex], result));

                AnalyticMeasurementDistribution measDist = calc.computeSignificance();

                System.out.printf("MI_Discrete(%s|%s) = %.4f bits (analytic p(surrogate > measured)=%.5f)\n",
                        variableNames[sourceIndex], variableNames[destIndex], result, measDist.pValue);
            }

            int finalSourceIndex = sourceIndex;
            javax.swing.SwingUtilities.invokeLater(() -> {
                MutualInfo chart = new MutualInfo("Mutual Information Results (Source: " + variableNames[finalSourceIndex] + ")", mutualInfoResults, variableNames[finalSourceIndex]);
                chart.pack();
                chart.setVisible(true);
            });
        }

        mutualInfoResultsList.sort(Comparator.comparingDouble((MutualInfoResult r) -> r.mutualInfoValue).reversed());

        System.out.println("\nSorted Mutual Information Results (High to Low):");
        for (MutualInfoResult result : mutualInfoResultsList) {
            System.out.printf("MI_Discrete(%s|%s) = %.4f bits\n", result.source, result.destination, result.mutualInfoValue);
        }

        saveToCsv(mutualInfoResultsList);
    }

    public static void saveToCsv(List<MutualInfoResult> mutualInfoResultsList) {
        String filePath = "A:\\mutual_information_results_discrete.csv";
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("Source,Destination,Mutual Information (bits)");

            for (MutualInfoResult result : mutualInfoResultsList) {
                writer.printf("%s,%s,%.4f\n", result.source, result.destination, result.mutualInfoVal
