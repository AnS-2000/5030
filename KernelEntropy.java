import infodynamics.utils.ArrayFileReader;
import infodynamics.utils.MatrixUtils;
import infodynamics.measures.continuous.kernel.EntropyCalculatorKernel;

public class KernelEntropy {

    public static String[] variableName = {
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
    };

    public static void main(String[] args) throws Exception {
        String dataFile = "C:\\Users\\Eric\\Desktop\\iris\\encoded_adult_classification_c.data";
        calculateKernelEntropy(dataFile);
    }

    public static void calculateKernelEntropy(String dataFile) throws Exception {
        ArrayFileReader afr = new ArrayFileReader(dataFile);
        double[][] data = afr.getDouble2DMatrix();

        EntropyCalculatorKernel calc = new EntropyCalculatorKernel();
        calc.setProperty(EntropyCalculatorKernel.KERNEL_WIDTH_PROP_NAME, "1");

        System.out.println("Kernel-based Entropy Results:");

        for (int v = 0; v < variableName.length; v++) {
            double[] variable = MatrixUtils.selectColumn(data, v);

            calc.initialise();
            calc.setObservations(variable);
            double result = calc.computeAverageLocalOfObservations();

            System.out.printf("H_Kernel(%s) = %.4f bits\n", variableName[v], result);
        }
    }
}
