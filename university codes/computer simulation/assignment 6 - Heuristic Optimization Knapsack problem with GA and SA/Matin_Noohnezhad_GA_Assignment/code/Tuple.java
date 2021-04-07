import java.util.Random;

/**
 * This is a class for storing the information of a tuple. There are various
 * functions for different purposes in the Tuple class.
 *
 * @author MCS3108
 */
public class Tuple {

    // An array that stores a configuration
    int[] configuration;

    // A variable that stores the cost of a configuration
    double cost;

    static Random random = new Random();

    /**
     * @return the configuration
     */
    public int[] getConfiguration() {
        return configuration;
    }

    /**
     * @param configuration the configuration to set
     */
    public void setConfiguration(int[] configuration) {
        this.configuration = configuration;
    }

    /**
     * @return the cost
     */
    public double getCost() {
        return cost;
    }

    /**
     * @param cost the cost to set
     */
    public void setCost(double cost) {
        this.cost = cost;
    }

    /**
     * a function for calculating the objective function of each tuple
     *
     * @param tpl (an object that refers to a configuration)
     * @return cost (a double variable for storing the objective function)
     */
    public static double getObjectiveFunction(Tuple tpl) {
        int[] config = tpl.getConfiguration();
        double profit = 0;

        for (int i = 0; i < config.length; i++) {
            if (config[i] == 1)
                profit += GA_Algorithm.values[i];
        }
        return profit;
    }

    /**
     * A function for generating a random tuple.
     *
     * @param list
     * @return the generated tuple
     */
    public static Tuple randomTuple(int[] list) {
        int[] config = new int[list.length];
        random.setSeed(System.currentTimeMillis());
        double a = 0.5;
        boolean status = false;
        while (!status) {
            for (int i = 0; i < list.length; i++) {
                if (random.nextDouble() < a)
                    config[i] = 1;
                else
                    config[i] = 0;
            }
            if (GA_Algorithm.isValid(config))
                status = true;
        }
        Tuple tup = new Tuple();
        tup.setConfiguration(config);
        tup.setCost(Tuple.getObjectiveFunction(tup));
        return tup;
    }
}
