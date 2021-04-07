import java.io.IOException;

/***
 *
 * @author MCS3108
 *
 */

public class SA_Algorithm {

    /**
     * The initial temperature
     */
    private static final int INITIAL_TEMPRETURE = 5000;
    //    public static int n = 50; // number of goods
    public static int[] weights;
    public static int[] values;
    public static int knapsack_weight;
    public static double profit = 0;
    //    public static boolean asdf = true;
//    public static double sumOfSquares = 0;
//    public static double sumForAverage = 0;

    /**
     * The constant K
     */
    public static double K = 0.5;

    /**
     * The entry point of the program which contains the implementation of
     * Simulated Annealing algorithm.
     * <p>
     * //	 * @param args
     * main method arguments
     */
    public static void start() throws IOException {

        int[] config = createRandomInitialConfig();
        double E = objectiveFunction(config);
        int T = INITIAL_TEMPRETURE;
        int unChangedCounter = 1;
        while (!terminate(T, unChangedCounter)) {
            for (int i = 0; i < counterDefiner(T); i++) {
                int[] config1 = changeConfig(config);
                double E1 = objectiveFunction(config1);
                if (isValid(config1)
                        && (E1 > E || (Math.random() < boltzmannFunction(E, E1,
                        K, T)))) {
                    config = config1;
                    E = E1;
//                    SA_Coordinator_Part_I.chart_29_data_writer(E);
//                    System.out.println(E);
                    unChangedCounter = 0;
                }
//                sumForAverage += E;
//                sumOfSquares += Math.pow(E, 2);
            }
//            SA_Coordinator_Part_I.chart_30_data_writer(T, Math.abs((sumOfSquares / (double) (counterDefiner(T)) - Math.pow(sumForAverage / (double) (counterDefiner(T)), 2))));
//            sumOfSquares = 0;
//            sumForAverage = 0;

            T = reduce(T);
            if (unChangedCounter != 0)
                unChangedCounter++;
        }

        printConfiguration(config);
//        int w = 0;
//        int v = 0;
//        for (int i = 0; i < values.length; i++) {
//            if (config[i] == 1) {
//                w += weights[i];
//                v += values[i];
//            }
//        }
//        System.out.println("knapsack weight is: " + knapsack_weight);
//        System.out.println("total weight is: " + w);
//        System.out.println("total value is: " + v);
        System.out.println();
        System.out.println("Result: " + E);
//        profit = E;
    }

    /**
     * Creates a random configuration
     *
     * @return a random configuration
     */
    private static int[] createRandomInitialConfig() {

        int[] config = new int[values.length];
        double a = 0.5;
        boolean status = false;
        while (!status) {
            for (int i = 0; i < values.length; i++) {
                if (Math.random() > a) {
                    config[i] = 1;
                } else {
                    config[i] = 0;
                }
            }
            if (isValid(config))
                status = true;
        }
        return config;
    }

    /**
     * Computes the objective value of a configuration
     *
     * @param config a configuration which represents a potential answer to the
     *               problem
     * @return the objective value of a configuration
     */
    private static int objectiveFunction(int[] config) {

        int sumOfValues = 0;

        for (int i = 0; i < values.length; i++) {
            if (config[i] == 1) {
                sumOfValues += values[i];
            }
        }
        return sumOfValues;

    }

    /***
     * Implements the Boltzmann function
     *
     * @param e
     *            the best configuration result so far
     * @param e1
     *            the current temporary configuration result
     * @param k
     *            the temperature constant
     * @param t
     *            the current temperature
     * @return the Boltzmann function result
     */
    private static double boltzmannFunction(double e, double e1, double k, int t) {

        double power = -1 * k * Math.abs(e - e1) / t;

        return Math.exp(power);

    }

    /**
     * Defines the number of times we change the configuration in a temperature
     *
     * @param T is the temperature
     * @return
     */
    private static int counterDefiner(int T) {
//        if (asdf)
//            return 2 * T;
//        else
        return 100;
    }

    /**
     * Change the configuration a little by some function
     *
     * @param config the current configuration
     * @return the changed configuration
     */
    private static int[] changeConfig(int[] config) {

        int[] newConfig = config.clone();
        int rand = (int) (Math.random() * values.length);
        if (newConfig[rand] == 0)
            newConfig[rand] = 1;
        else
            newConfig[rand] = 0;
        return newConfig;

    }

    /**
     * Check to see if the current configuration is a valid configuration
     * according to the problem definition
     *
     * @param config the current configuration
     * @return true if configuration is valid and false otherwise
     */
    private static boolean isValid(int[] config) {

        int sumOfWeights = 0;
        for (int i = 0; i < values.length; i++) {
            if (config[i] == 1) {
                sumOfWeights += weights[i];
            }
        }
        if (sumOfWeights > knapsack_weight)
            return false;
        else
            return true;
    }

    /**
     * @param T                current temperature
     * @param unChangedCounter number of times that the objective value did not changed
     * @return boolean showing if algorithm should be terminated
     */
    private static boolean terminate(int T, int unChangedCounter) {

        if (T == 0 || unChangedCounter == 100)
            return true;
        return false;
    }

    /**
     * Reduces the current temperature by some function
     *
     * @param T the current temperature
     * @return the reduced temperature
     */
    private static int reduce(int T) {
        int temp = T * 19 / 20;
        if (temp == T)
            return T--;
        else
            return temp;
    }

    private static void printConfiguration(int[] config) {
        String indexString = "";
        String configString = "";
        for (int i = 0; i < config.length; i++) {
            indexString += " " + i;
            configString += " " + config[i];
        }
        System.out.println("Index:         " + indexString);
        System.out.println("Configuration: " + configString);
    }
}
