import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

/**
 * A class for generating data for drawing (Mutation probability - Executation
 * time), (Crossover probability - Executation time) and (Population size -
 * Executation time) curves for n = 50, 500 and 5000
 *
 * @author MCS3108
 */
public class GA_Coordinator_Part_I {

    // The maximum size of populations

    static final int MAX_POP_SIZE = 500;

    static final double MAX_MUTATION_PROBABILITY = 0.3;

    static final double MAX_CROSSOVER_PROBABILITY = 0.9;

    // total number of repeats in each configuration size and mutation/crossover
    // probability

    static final int MAX_REPEAT = 10;

    /**
     * this variable should be true when you want to generate data for (
     * Mutation probability - Executation time) curve, otherwise it should be
     * false.
     */

    public static boolean mutation = true;

    /**
     * this variable should be true when you want to generate data for (
     * Crossover probability - Executation time) curve, otherwise it should be
     * false.
     */

    public static boolean crossover = false;

    /**
     * this variable should be true when you want to generate data for (
     * population size - Executation time) curve, otherwise it should be false.
     */

    public static boolean population = false;

    public static void main(String[] args) throws IOException {

//        chart_1();
//        chart_2();
//        chart_7();
//        chart_13();
    }

    public static void chart_1() throws IOException {
        FileWriter rep = new FileWriter("q1_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("Population Size , Average Profit , ");
        bw.newLine();
        double[] sumProfitArray = new double[MAX_POP_SIZE+1 ];
        for (int j = 0; j < MAX_REPEAT; j++) {
            System.out.println("repeat" + j);
            GA_Algorithm.getData();
            for (int i = 20; i <= MAX_POP_SIZE; i += 10) {
                System.out.println(i);
                GA_Algorithm.RWeels = new double[i];
                GA_Algorithm.POPULATION_SIZE = i;
//                System.out.println(i);
                GA_Algorithm.counter = 1;
                GA_Algorithm.repeats = 0;
                GA_Algorithm.firstPop = Population.randomPopulation(GA_Algorithm.list);
                sumProfitArray[i] += GA_Algorithm.genetic_function();
            }
        }
        for (int i = 20; i <= MAX_POP_SIZE; i += 10) {
            NumberFormat formatter = new DecimalFormat("#0.00");
            bw.write(i + " , " + formatter.format(sumProfitArray[i] / MAX_REPEAT) + " , ");
            bw.newLine();
        }
        //        for (int i = 20; i <= MAX_POP_SIZE; i += 2) {
//            System.out.println(i);
//            GA_Algorithm.POPULATION_SIZE = i;
//            GA_Algorithm.RWeels = new double[i];
//            double sumForAverage = 0;
//            for (int j = 0; j < MAX_REPEAT; j++) {
//                sumForAverage += GA_Algorithm.genetic_function();
//            }
//            NumberFormat formatter = new DecimalFormat("#0.00");
//            bw.write(i + " , " + formatter.format(sumForAverage / MAX_REPEAT) + " , ");
//            bw.newLine();
//        }
        bw.close();
        rep.close();
    }

    public static void chart_2() throws IOException {
//        GA_Algorithm.getData();
        FileWriter rep = new FileWriter("q2_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("Population Size , Execution Time , ");
        bw.newLine();
        double[] sumExecutionTime = new double[MAX_POP_SIZE+1 ];
        for (int j = 0; j < MAX_REPEAT; j++) {
            System.out.println("repeat: " +j);
            GA_Algorithm.getData();
            for (int i = 20; i <= MAX_POP_SIZE; i += 10) {
                System.out.println(i);
                GA_Algorithm.RWeels = new double[i];
                GA_Algorithm.POPULATION_SIZE = i;
//                System.out.println(i);
                GA_Algorithm.counter = 1;
                GA_Algorithm.repeats = 0;
                GA_Algorithm.firstPop = Population.randomPopulation(GA_Algorithm.list);
                double beforeTime = System.currentTimeMillis();
                GA_Algorithm.genetic_function();
                double afterTime = System.currentTimeMillis();
                sumExecutionTime[i] += (afterTime - beforeTime);
            }
        }
        for (int i = 20; i <= MAX_POP_SIZE; i += 10) {
            NumberFormat formatter = new DecimalFormat("#0.00");
            bw.write(i + " , " + formatter.format(sumExecutionTime[i] / MAX_REPEAT) + " , ");
            bw.newLine();
        }
//        for (int i = 20; i <= MAX_POP_SIZE; i += 2) {
//            System.out.println(i);
//            GA_Algorithm.POPULATION_SIZE = i;
//            GA_Algorithm.RWeels = new double[i];
//            double sumTime = 0;
//
//            for (int j = 0; j < MAX_REPEAT; j++) {
//                double beforeTime = System.currentTimeMillis();
//                GA_Algorithm.genetic_function();
//                double afterTime = System.currentTimeMillis();
//                sumTime += (afterTime - beforeTime);
//            }
//            NumberFormat formatter = new DecimalFormat("#0.00");
//            bw.write(i + " , " + formatter.format(sumTime / MAX_REPEAT) + " , ");
//            bw.newLine();
//        }
        bw.close();
        rep.close();
    }

    public static void chart_7() throws IOException {
        FileWriter rep = new FileWriter("q7_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("Mutation Probability , Average Profit , ");
        bw.newLine();
        double[] sumProfitArr = new double[(int)(MAX_MUTATION_PROBABILITY*1000)+1];
        for(int j =0;j<MAX_REPEAT;j++){
            GA_Algorithm.getData();
            for(double i =0.005;i<=MAX_MUTATION_PROBABILITY;i+=0.005){
                System.out.println(i);
                GA_Algorithm.MUTATION_PROBABILITY = i;
                GA_Algorithm.counter = 1;
                GA_Algorithm.repeats = 0;
                sumProfitArr[(int)(i*1000)] += GA_Algorithm.genetic_function();
            }
        }
        for(double i =0.005;i<=MAX_MUTATION_PROBABILITY;i+=0.005){
            NumberFormat formatter = new DecimalFormat("#0.00");
            bw.write(i + " , " + formatter.format(sumProfitArr[(int)(i*1000)] / MAX_REPEAT) + " , ");
            bw.newLine();
        }
//        for (double i = 0.005; i <= MAX_MUTATION_PROBABILITY; i += 0.005) {
//            System.out.println(i);
//            GA_Algorithm.MUTATION_PROBABILITY = i;
//            double sumProfit = 0;
//            for (int j = 0; j < MAX_REPEAT; j++) {
//                GA_Algorithm.counter = 1;
//                GA_Algorithm.repeats = 0;
//                sumProfit += GA_Algorithm.genetic_function();
//            }
//            NumberFormat formatter = new DecimalFormat("#0.00");
//            bw.write(i + " , " + formatter.format(sumProfit / MAX_REPEAT) + " , ");
//            bw.newLine();
//        }
        bw.close();
        rep.close();
    }

    public static void chart_13() throws IOException {
        GA_Algorithm.getData();
        FileWriter rep = new FileWriter("q13_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("Crossover Probability , Average Profit , ");
        bw.newLine();
        double[] sumProfitArr = new double[(int)((MAX_CROSSOVER_PROBABILITY - 0.1)/0.05)+1];
        for(int j =0;j<MAX_REPEAT;j++){
            GA_Algorithm.getData();
            for(double i =0.1;i<=MAX_CROSSOVER_PROBABILITY;i+=0.05){
                System.out.println(i);
                GA_Algorithm.CROSSOVER_PROBABILITY= i;
                GA_Algorithm.counter = 1;
                GA_Algorithm.repeats = 0;
                sumProfitArr[(int)((i - 0.1)/0.05)] += GA_Algorithm.genetic_function();
            }
        }
        for(double i =0.1;i<=MAX_CROSSOVER_PROBABILITY;i+=0.05){
            NumberFormat formatter = new DecimalFormat("#0.00");
            bw.write(i + " , " + formatter.format(sumProfitArr[(int)((i - 0.1)/0.05)] / MAX_REPEAT) + " , ");
            bw.newLine();
        }
//        for (double i = 0.1; i <= MAX_CROSSOVER_PROBABILITY; i += 0.1) {
//            System.out.println(i);
//            GA_Algorithm.CROSSOVER_PROBABILITY = i;
//            double sumProfit = 0;
//            for (int j = 0; j < MAX_REPEAT; j++) {
//                sumProfit += GA_Algorithm.genetic_function();
//            }
//            NumberFormat formatter = new DecimalFormat("#0.00");
//            bw.write(i + " , " + formatter.format(sumProfit / MAX_REPEAT) + " , ");
//            bw.newLine();
//        }
        bw.close();
        rep.close();
    }

    public static void chart_19(int generation_no, double profit) throws IOException {
        System.out.println(generation_no + "  ,  " + profit);
        FileWriter rep = new FileWriter("q19_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        NumberFormat formatter = new DecimalFormat("#0.00");
        bw.write(generation_no + " , " + formatter.format(profit) + " , ");
        bw.newLine();
        bw.close();
        rep.close();
    }

    public static void chart_20(int generation_no, double variance) throws IOException {
        System.out.println(generation_no + "  ,  " + variance);
        FileWriter rep = new FileWriter("q20_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        NumberFormat formatter = new DecimalFormat("#0.00");
        bw.write(generation_no + " , " + formatter.format(variance) + " , ");
        bw.newLine();
        bw.close();
        rep.close();
    }

//    public static void main(String args[]) {
//
//        // variables for computing the running time of the algorithm
//        long time, tempTime;
//
//        System.out
//                .println("config, mutation/crossover probability/population size, time");
//
//        // a loop in which the size of configuration is increasing
//        for (int configSize = 50; configSize <= 5000; configSize *= 10) {
//
//            if (mutation) {
//
//                // a loop in which the size of the configuration is fixed
//                // but the mutation probability is increasing
//                for (double k = 0.001; k < MAX_MUTATION_PROBABILITY; k += 0.001) {
//
//                    time = 0;
//
//                    // a loop in which both configurarion size and the mutation
//                    // probability are
//                    // fixed, but the algorithm is repeated.
//
//                    for (int i = 0; i < MAX_REPEAT; i++) {
//
//                        GA_Algorithm.list = new int[configSize];
//                        generateRandomData(configSize);
//                        tempTime = System.currentTimeMillis();
//                        GA_Algorithm.MUTATION_PROBABILITY = k;
//                        GA_Algorithm.genetic_function();
//                        tempTime = System.currentTimeMillis() - tempTime;
//                        time += tempTime;
//                    }
//                    time /= MAX_REPEAT;
//                    System.out.println(configSize + "," + k + "," + time);
//                }
//            } else if (crossover) {
//
//                // a loop in which the size of the configuration is fixed
//                // but the crossover probability is increasing
//
//                for (double j = 0.1; j < MAX_CROSSOVER_PROBABILITY; j += 0.1) {
//                    time = 0;
//
//                    // a loop in which both configurarion size and the crossover
//                    // probability are
//                    // fixed, but the algorithm is repeated.
//
//                    for (int i = 0; i < MAX_REPEAT; i++) {
//
//                        GA_Algorithm.list = new int[configSize];
//
//                        generateRandomData(configSize);
//                        tempTime = System.currentTimeMillis();
//                        GA_Algorithm.CROSSOVER_PROBABILITY = j;
//                        GA_Algorithm.genetic_function();
//                        tempTime = System.currentTimeMillis() - tempTime;
//                        time += tempTime;
//                    }
//                    time /= MAX_REPEAT;
//                    System.out.println(configSize + "," + j + "," + time);
//                }
//            }
//            if (population) {
//
//                // a loop in which the size of the configuration is fixed
//                // but the population size is increasing
//
//                int temp = 10;
//                int counter = 0;
//                for (int j = 20; j < MAX_POP_SIZE; j += temp) {
//                    counter++;
//                    time = 0;
//
//                    // a loop in which both configurarion size and the
//                    // population size
//                    // are fixed, but the algorithm is repeated.
//
//                    for (int i = 0; i < MAX_REPEAT; i++) {
//
//                        GA_Algorithm.list = new int[configSize];
//
//                        generateRandomData(configSize);
//                        tempTime = System.currentTimeMillis();
//                        GA_Algorithm.POPULATION_SIZE = j;
//                        GA_Algorithm.genetic_function();
//                        tempTime = System.currentTimeMillis() - tempTime;
//                        time += tempTime;
//                    }
//                    time /= MAX_REPEAT;
//                    System.out.println(configSize + "," + j + "," + time);
//                    if (counter == 9)
//                        temp = 100;
//                }
//            }
//        }
//
//    }

    /**
     * Depending on your problem, generate random integers/doubles and store
     * them in the static lists you've defined in the GA_Algorithm class.
     * <p>
     * for example, if you have the OBST problem, you must generate random data
     * ,probabilitis and failures and store them in the corresponding lists.
     *
     * @param configSize
     */
    public static void generateRandomData(int configSize) {

        GA_Algorithm.knapsack_weight = (5 + (int) (Math.random() * 11)) * configSize;//random size between 5*n to 15*n
//        System.out.println(SA_Algorithm.knapsack_weight);
        GA_Algorithm.weights = new int[configSize];
        GA_Algorithm.values = new int[configSize];
        for (int i = 0; i < configSize; i++) {

            GA_Algorithm.weights[i] = 1 + (int) (Math.random() * 50);//random number between 1 to 50
            GA_Algorithm.values[i] = 1 + (int) (Math.random() * 100);//random number between 1 to 100
        }
        GA_Algorithm.firstPop = Population.randomPopulation(GA_Algorithm.list);

    }
}