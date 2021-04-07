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
    private static final int INITIAL_TEMPRETURE = 100;
    //    public static int n = 50; // number of goods
//    public static int[] weights;
//    public static int[] values;
//    public static int knapsack_weight;
    public static double cost_coefficient1;
    public static double cost_coefficient2;
    public static double max_budget;
    public static double profit = 0;
    public static int lambda;
    public static int delta;
    public static int mio;
    public static int simulationTime;
    public static int[] config;
    //    public static boolean asdf = true;
//    public static double sumOfSquares = 0;
//    public static double sumForAverage = 0;

    /**
     * The constant K
     */
    public static double K = 0.005;

    /**
     * The entry point of the program which contains the implementation of
     * Simulated Annealing algorithm.
     * <p>
     * //	 * @param args
     * main method arguments
     */
    public static void start() throws IOException {

        config = createRandomInitialConfig();
        double E = objectiveFunction(config[0], config[1]);
        int T = INITIAL_TEMPRETURE;
        int unChangedCounter = 1;
        while (!terminate(T, unChangedCounter)) {
            System.out.println(max_budget +" , "+ T);
//            printConfiguration(config);
            for (int i = 0; i < counterDefiner(T); i++) {
                int[] config1 = changeConfig(config);
                double E1 = objectiveFunction(config1[0], config1[1]);
                if (isValid(config1)
                        && (E1 < E || (Math.random() < boltzmannFunction(E, E1,
                        K, T)))) {
                    config = config1;
                    E = E1;
                    profit = E;
                    unChangedCounter = 0;
                }
            }

            T = reduce(T);
            if (unChangedCounter != 0) {
                unChangedCounter++;
            }
        }

        printConfiguration(config);
    }

    /**
     * Creates a random configuration
     *
     * @return a random configuration
     */
    private static int[] createRandomInitialConfig() {
        int[] config = new int[2];
        boolean status = false;
        while (!status) {
            config[0] = 1 + (int) (Math.random() * 10);
            config[1] = 1 + (int) (Math.random() * 10);
            if (config[0] * cost_coefficient1 + config[1] * cost_coefficient2 <= max_budget) {
                status = true;
            }
        }
        return config;
    }

    /**
     * Computes the objective value of a configuration
     * <p>
     * //     * @param config a configuration which represents a potential answer to the
     * problem
     *
     * @return the objective value of a configuration
     */
    private static double objectiveFunction(int noQueues, int noServer) {
        return new Executer(simulationTime, noQueues, noServer, lambda, delta, mio).simualte();
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
        return 10;
    }

    /**
     * Change the configuration a little by some function
     *
     * @param config the current configuration
     * @return the changed configuration
     */
    private static int[] changeConfig(int[] config) {
        boolean status = false;
        int[] newConfig = config.clone();
        while (!status) {
            int rand = (int) (Math.random() * 2);
            if (rand == 2) {
                rand = 1;
            }
            if (rand == 0) {
                if (newConfig[0] == 1)
                    newConfig[0]++;
                else {
                    if ((Math.random() * 2) < 1) {
                        newConfig[0]--;
                    } else {
                        newConfig[0]++;
                    }
                }
            } else {
                if (newConfig[1] == 1)
                    newConfig[1]++;
                else {
                    if ((Math.random() * 2) < 1) {
                        newConfig[1]--;
                    } else {
                        newConfig[1]++;
                    }
                }
            }
            if (isValid(newConfig))
                status = true;
            else
                newConfig = config.clone();
        }
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
        if (config[0] * cost_coefficient1 + config[1] * cost_coefficient2 <= max_budget)
            return true;
        else
            return false;
    }

    /**
     * @param T                current temperature
     * @param unChangedCounter number of times that the objective value did not changed
     * @return boolean showing if algorithm should be terminated
     */
    private static boolean terminate(int T, int unChangedCounter) {

        if (T == 0 || unChangedCounter == 20)
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
        System.out.println("used budget is: " + (config[0] * cost_coefficient1 + config[1] * cost_coefficient2));
        System.out.println("max budget is: " + max_budget);
        System.out.println("average waiting time is: " + profit);
    }
}
