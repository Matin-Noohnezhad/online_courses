import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * @author MCS3108
 */

public class GA_Algorithm {

    public static int[] weights;
    public static int[] values;
    public static int knapsack_weight;
    public static int number = 50; //size of knapsack
    public static Population firstPop;
//    public static int profit = 0;

    static Random random = new Random();

    public static int[] list;

    // This variable stores the population size
    public static int POPULATION_SIZE = 20;

    // This variable stores the mutation probability constant
    public static double MUTATION_PROBABILITY = 0.2;

    // This variable stores the mutation probability constant
    public static double CROSSOVER_PROBABILITY = 0.2;

    // This variable stores the number of populations
    public static int POPULATION_NUMBER = 2500;

    // an integer variable that stores the number of created populations.
    // It will incremented by one when a new population is generated.
    static int counter = 1;

    // a double variable that stores the number of consecutive populations that
    // their
    // minimum cost has not improved.
    static double repeats = 0;

    // An array for storing the values of Roulette Weels.
    static double[] RWeels = new double[POPULATION_SIZE];
    ;

    public static double genetic_function() throws IOException {

        // get data from user
//        getData();
        // Create initial population of chromosomes
//        Population pop = Population.randomPopulation(list);
        Population pop = firstPop;
        // Sort the chromosomes of first population based on their cost.
        Population.sortByCost(pop.chromosomes);
        // Create Roulette Weels
//        createRouletteWeels(pop);
        // repeat following steps until stop criterion is satisfied.
        GA_Coordinator_Part_I.chart_20(counter , variance(pop));
        GA_Coordinator_Part_I.chart_19(counter, pop.chromosomes.get(0).cost);
        while (!terminated()) {
//            System.out.println(POPULATION_SIZE);
//            System.out.println(pop.getChromosomes().size());
            createRouletteWeels(pop);

            // Generate next population
            Population nextPop = selection(pop);
            counter++;
            // Sort the chromosomes of new population based on their cost.
            Population.sortByCost(nextPop.chromosomes);
            // If the optimal cost has not improved increment "repeats" by one
            // otherwise, set its value to 0.
            if (nextPop.chromosomes.get(0).cost <= pop.chromosomes
                    .get(0).cost) {
                repeats++;
            } else {
                repeats = 0;
            }

            for (int i = 0; i < pop.chromosomes.size(); i++) {
                pop.chromosomes.set(i, nextPop.chromosomes.get(i));
                pop.chromosomes.get(i).cost = nextPop.chromosomes.get(i).cost;
            }
            Population.sortByCost(pop.chromosomes);
            GA_Coordinator_Part_I.chart_20(counter, variance(nextPop));
            GA_Coordinator_Part_I.chart_19(counter, nextPop.chromosomes.get(0).cost);
        }

        // Output the best solution found
//		return pop.chromosomes.get(pop.chromosomes.size() - 1).cost;
        Population.sortByCost(pop.chromosomes);
//        printConfiguration(pop.chromosomes.get(0).getConfiguration());
        int[] config = pop.chromosomes.get(0).getConfiguration();
//        System.out.println("knapsack weight: " + knapsack_weight);
//        double sumWeight = 0;
//        double sumValue = 0;
//        for (int i = 0; i < values.length; i++) {
//            if (config[i] == 1) {
//                sumWeight += weights[i];
//                sumValue += values[i];
//            }
//        }
//        System.out.println("sum weights: " + sumWeight);
//        System.out.println("sum values: " + sumValue);
        return pop.chromosomes.get(0).cost;
    }

    /**
     * a function that selects chromosomes from the current population to be
     * parents to crossover.
     *
     * @param pop (an object that refers to the current population)
     * @return the next population
     */

    private static Population selection(Population pop) {

        Population nextPop = new Population();
        // An array that stores the two best configurations of current
        // population
        Tuple[] pr = selectTheTwoBest(pop);
        // Add the two best configurations to the next population
        nextPop.chromosomes.add(pr[0]);
        nextPop.chromosomes.add(pr[1]);

        // ***************************

        for (int k = 0; k < (GA_Algorithm.POPULATION_SIZE / 2) - 1; k++) {

            // Select two chromosomes from the previous population

            Tuple tpl1 = selectChromosome(pop);
            Tuple tpl2 = selectChromosome(pop);
            // check for crossover on selected chromosomes
            if (random.nextDouble() < GA_Algorithm.CROSSOVER_PROBABILITY) {

                Tuple[] pair = crossover(tpl1, tpl2);

                // check for mutation on each of selected chromosomes
                if (random.nextDouble() < GA_Algorithm.MUTATION_PROBABILITY) {
                    Tuple tp = mutation(pair[0]);
                    tp.cost = Tuple.getObjectiveFunction(tp);
                    nextPop.chromosomes.add(tp);
                } else {
                    Tuple tp = pair[0];
                    tp.cost = Tuple.getObjectiveFunction(tp);
                    nextPop.chromosomes.add(tp);
                }
                if (random.nextDouble() < GA_Algorithm.MUTATION_PROBABILITY) {
                    Tuple tp1 = mutation(pair[1]);
                    tp1.cost = Tuple.getObjectiveFunction(tp1);
                    nextPop.chromosomes.add(tp1);
                } else {
                    Tuple tp1 = pair[1];
                    tp1.cost = Tuple.getObjectiveFunction(tp1);
                    nextPop.chromosomes.add(tp1);

                }

            } else {
                // If no crossover was performed, add the selected chromosomes
                // to the next population.
                nextPop.chromosomes.add(tpl1);
                nextPop.chromosomes.add(tpl2);
            }
        }
        return nextPop;

    }

    /**
     * A function for doing mutation.
     *
     * @param tuple: an object that refers to a tuple
     * @return the tupple after mutation
     */
    private static Tuple mutation(Tuple tuple) {
        int size = tuple.getConfiguration().length;
        int[] config = tuple.getConfiguration();
        boolean status = false;
        while (!status) {
            random.setSeed(System.currentTimeMillis());
            int index = (int) (random.nextDouble() * (size));
            if (config[index] == 1)
                config[index] = 0;
            else
                config[index] = 1;
            if (isValid(config))
                status = true;
            else {
                if (config[index] == 1)
                    config[index] = 0;
                else
                    config[index] = 1;
            }
        }
        tuple.setConfiguration(config);
        return tuple;
    }

    /**
     * a function for doing crossover.
     *
     * @param tpl1: an object that refers to a tuple. It is one of the chromosomes
     *              that have been selected in the selection function.
     * @param tpl2: an object that refers to a tuple. It is one of the chromosomes
     *              that have been selected in the selection function.
     * @return pair: an array that stores the two tuples after crossover.
     */
    private static Tuple[] crossover(Tuple tpl1, Tuple tpl2) {
        random.setSeed(System.currentTimeMillis());
        int size = tpl1.getConfiguration().length;
        Tuple[] pair = new Tuple[2];
        Tuple t1 = new Tuple();
        Tuple t2 = new Tuple();
        int[] a1 = new int[size];
        int[] a2 = new int[size];
        boolean status = false;
        while (!status) {
            int index = (int) (random.nextDouble() * (size));
            int i = 0;
            for (; i < index; i++) {
                a1[i] = tpl1.getConfiguration()[i];
                a2[i] = tpl2.getConfiguration()[i];
            }
            for (; i < size; i++) {
                a1[i] = tpl2.getConfiguration()[i];
                a2[i] = tpl1.getConfiguration()[i];
            }
            if (isValid(a1) && isValid(a2))
                status = true;
        }
        t1.setConfiguration(a1);
        t2.setConfiguration(a2);
        t1.setCost(Tuple.getObjectiveFunction(t1));
        t2.setCost(Tuple.getObjectiveFunction(t2));
        pair[0] = t1;
        pair[1] = t2;
        return pair;
    }

    /**
     * This function selects a chromosome from current population using roulette
     * weels selection
     *
     * @param pop: current population
     * @return the selected chromosome
     */
    private static Tuple selectChromosome(Population pop) {
        random.setSeed(System.currentTimeMillis());
        double a = random.nextDouble();
        Tuple t = null;
        boolean status = false;
        int count = 0;
        while (!status) {
            if (a < RWeels[count]) {
                t = pop.getChromosomes().get(count);
                status = true;
            }
            count++;
        }
        if (t == null)
            System.out.println("roulette wheel code is wrong!!");
        return t;
    }

    /**
     * A function for creating Roulette Weels. By using Roulette Weels
     * selection, the better chromosomes have more chance to be selected. Store
     * the values of Roulette Weels in the "RWeels" array
     */
    private static void createRouletteWeels(Population pop) {

        double sumCost = 0;
//        for (int i = 0; i < pop.getChromosomes().size(); i++) {
        for (int i = 0; i < POPULATION_SIZE; i++) {
            sumCost += pop.getChromosomes().get(i).getCost();
            RWeels[i] = sumCost;
        }
        for (int i = 0; i < pop.getChromosomes().size(); i++) {
            RWeels[i] = RWeels[i] / sumCost;
        }

    }

    /**
     * A function for selecting two best chromosomes from current population
     *
     * @param pop :
     *            current population
     * @return an array that stores the two selected chromosomes
     */
    private static Tuple[] selectTheTwoBest(Population pop) {
        Tuple[] twoTup = new Tuple[2];
        ArrayList<Tuple> chromosomes = pop.getChromosomes();
        Population.sortByCost(chromosomes);
        twoTup[0] = chromosomes.get(0);
        twoTup[1] = chromosomes.get(1);
        return twoTup;
    }

    /**
     * a function that specifies the end of genetic algorithm.
     * <p>
     * end criterion: if the number of created populations is equal with
     * "POPULATION_NUMBER", or if for 50 consecutive populations, the objective
     * function doesn't improved.
     *
     * @return a boolean. If the end condition is satified the return value is
     * "true" otherwise, it's "false".
     */

    private static boolean terminated() {
        if (counter == POPULATION_NUMBER || repeats == 500)
            return true;
        return false;
    }

    //from sa algorithm code
    public static boolean isValid(int[] config) {
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
     * A function for getting data from user
     */
    public static void getData() {
//        Scanner scan = new Scanner(System.in);
//        System.out.println("Enter the number of nodes/jobs/cities: ");
        // variable for storing the number of nodes/jobs/cities
//        int number = scan.nextInt();
        // provide the demension of array using the number

        list = new int[number];

        /**
         * Complete the function
         *
         */
        GA_Coordinator_Part_I.generateRandomData(number);
//        double a = 0.5;
//        random.setSeed(System.currentTimeMillis());
//        for (int i = 0; i < number; i++) {
//            if (random.nextDouble() > a)
//                list[i] = 1;
//            else
//                list[i] = 0;
//        }
    }


    public static double variance(Population pop) {
        double sumProfits = 0;
        double sumVarProf = 0;
        for (int i = 0; i < pop.getChromosomes().size(); i++) {
            sumProfits += pop.getChromosomes().get(i).getCost();
            sumVarProf += Math.pow(pop.getChromosomes().get(i).getCost(), 2);
        }
        double avg = sumProfits / pop.getChromosomes().size();
        return Math.abs((sumVarProf / pop.getChromosomes().size()) - Math.pow(avg, 2));
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
