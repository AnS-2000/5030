import infodynamics.utils.ArrayFileReader;
import infodynamics.utils.EmpiricalMeasurementDistribution;
import infodynamics.utils.MatrixUtils;
import infodynamics.measures.discrete.ConditionalMutualInformationCalculatorDiscrete;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class MutualInfoCond extends JFrame {

    private static final String[] variableNames = {
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
    };

    static class ConditionalMutualInfoResult {
        String source;
        String destination;
        String conditional;
        double conditionalMutualInfoValue;

        public ConditionalMutualInfoResult(String source, String destination, String conditional, double value) {
            this.source = source;
            this.destination = destination;
            this.conditional = conditional;
            this.conditionalMutualInfoValue = value;
        }
    }

    public MutualInfoCond(String title, List<ConditionalMutualInfoResult> results) {
        super(title);

        DefaultCategoryDataset dataset = createDataset(results);

        JFreeChart chart = ChartFactory.createBarChart(
                title,
                "Source|Destination|Conditional",
                "Conditional Mutual Information (bits)",
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

    private DefaultCategoryDataset createDataset(List<ConditionalMutualInfoResult> results) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (ConditionalMutualInfoResult result : results) {
            String label = result.source + "|" + result.destination + "|" + result.conditional;
            dataset.addValue(result.conditionalMutualInfoValue, "Conditional Mutual Information", label);
        }
        return dataset;
    }

    public static void main(String[] args) throws Exception {
        String dataFile = "C:\\Users\\Eric\\Desktop\\iris\\encoded_adult_classification.data";
        ArrayFileReader afr = new ArrayFileReader(dataFile);
        int[][] data = afr.getInt2DMatrix();
        int[] Cvariables = {4, 9, 4, 16, 4, 7, 15, 6, 5, 2, 2, 5, 4, 42, 2};

        List<ConditionalMutualInfoResult> conditionalMutualInfoResultsList = new ArrayList<>();

        for (int sourceIndex = 0; sourceIndex < data[0].length; sourceIndex++) {
            int[] source = MatrixUtils.selectColumn(data, sourceIndex);

            for (int destIndex = sourceIndex + 1; destIndex < data[0].length; destIndex++) {
                int[] destination = MatrixUtils.selectColumn(data, destIndex);

                for (int condIndex = 0; condIndex < data[0].length; condIndex++) {
                    if (condIndex == sourceIndex || condIndex == destIndex) continue;

                    int[] conditional = MatrixUtils.selectColumn(data, condIndex);

                    ConditionalMutualInformationCalculatorDiscrete calc = new ConditionalMutualInformationCalculatorDiscrete(
                            Cvariables[sourceIndex], Cvariables[destIndex], Cvariables[condIndex]);

                    calc.initialise();
                    calc.addObservations(source, destination, conditional);
                    double result = calc.computeAverageLocalOfObservations();

                    conditionalMutualInfoResultsList.add(new ConditionalMutualInfoResult(
                            variableNames[sourceIndex], variableNames[destIndex], variableNames[condIndex], result));

                    System.out.printf("CMI_Discrete(%s -> %s | %s) = %.4f bits\n",
                            variableNames[sourceIndex], variableNames[destIndex], variableNames[condIndex], result);
                }
            }
        }

        conditionalMutualInfoResultsList.sort(Comparator.comparingDouble((ConditionalMutualInfoResult r) -> r.conditionalMutualInfoValue).reversed());

        System.out.println("\nSorted Conditional Mutual Information Results (High to Low):");
        for (ConditionalMutualInfoResult result : conditionalMutualInfoResultsList) {
            System.out.printf("CMI_Discrete(%s -> %s | %s) = %.4f bits\n",
                    result.source, result.destination, result.conditional, result.conditionalMutualInfoValue);
        }

        javax.swing.SwingUtilities.invokeLater(() -> {
            MutualInfoCond chart = new MutualInfoCond("Sorted Conditional Mutual Information Results", conditionalMutualInfoResultsList);
            chart.pack();
            chart.setVisible(true);
        });
    }
}
