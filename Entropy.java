import infodynamics.utils.ArrayFileReader;
import infodynamics.utils.MatrixUtils;
import infodynamics.measures.discrete.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import javax.swing.JFrame;

public class Entropy extends JFrame {
    public String[] getVariableName() {
        return variableName;
    }

    public static String[] variableName = {"age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"};

    public Entropy(String title, double[] results) {
        super(title);

        DefaultCategoryDataset dataset = createDataset(results, variableName);

        JFreeChart chart = ChartFactory.createLineChart(
                "Entropy Results",
                "Variable Index",
                "Entropy (bits)",
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

    private DefaultCategoryDataset createDataset(double[] results, String[] variableName) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (int i = 0; i < results.length; i++) {
            dataset.addValue(results[i], "Entropy", variableName[i]);
        }
        return dataset;
    }

    public static void main(String[] args) throws Exception {
        String dataFile = "C:\\Users\\Eric\\Desktop\\iris\\encoded_adult_classification.data";
        ArrayFileReader afr = new ArrayFileReader(dataFile);
        int[][] data = afr.getInt2DMatrix();
        int[] Cvariables = {4, 9, 4, 16, 4, 7, 15, 6, 5, 2, 2, 5, 4, 42, 2};

        EntropyCalculatorDiscrete calc = new EntropyCalculatorDiscrete(43);

        double[] results = new double[15];

        for (int v = 0; v < 15; v++) {
            int[] variable = MatrixUtils.selectColumn(data, v);

            calc.initialise(Cvariables[v]);
            calc.addObservations(variable);
            double result = calc.computeAverageLocalOfObservations();
            results[v] = result;

            System.out.printf("H_Discrete(%s) = %.4f bits\n", variableName[v], result);
        }

        javax.swing.SwingUtilities.invokeLater(() -> {
            Entropy chart = new Entropy("Entropy Results", results);
            chart.pack();
            chart.setVisible(true);
        });
    }
}
