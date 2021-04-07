import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class SA_Coordinator_Part_I {

    /**
     * The maximum value for K constant
     */
    static final int MAX_K = 5;

    /**
     * The number of times we repeat the algorithm for a same configuration
     */
    static final int MAX_REPEAT = 5;

//    public static void main(String args[]) {
//
//        /**
//         * Variables to compute time
//         */
//        long time, tempTime;
//
//        System.out.println("config, K constant, time");
//
//        // A loop in which the size of configuration is increasing
//        for (int configSize = 50; configSize <= 5000; configSize *= 10) {
//
//            // A loop in which the size of the configuration is fixed
//            // but the value of k is increasing
//            for (double k = 0.5; k < MAX_K; k += 0.5) {
//                time = 0;
//
//                // A loop in which both configuration and K are
//                // fixed, but the algorithm is repeating
//                for (int i = 0; i < MAX_REPEAT; i++) {
//
//                    // Your algorithm list(s) will be filled here
//                    // SA_Algorithm.list = new int[configSize];
//                    generateRandomData(configSize);
//                    tempTime = System.currentTimeMillis();
//                    SA_Algorithm.K = k;
//                    SA_Algorithm.start();
//                    tempTime = System.currentTimeMillis() - tempTime;
//                    time += tempTime;
//                }
//                time /= MAX_REPEAT;
//                System.out.println(configSize + "," + k + "," + time);
//            }
//        }
//    }

    public static void main(String[] args) throws IOException {
        chart_23();

    }

    public static void chart_23() throws IOException {
        FileWriter rep = new FileWriter("q23_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        NumberFormat formatter = new DecimalFormat("#0.00");
        bw.write("K , Average Profit,");
        bw.newLine();
        generateRandomData(50);
        for (double k = 0.5; k <= MAX_K; k += 0.5) {
            double sumProfit = 0;
            SA_Algorithm.K = k;
            for (int i = 0; i < MAX_REPEAT; i++) {
                SA_Algorithm.start();
                sumProfit += SA_Algorithm.profit;
            }
            bw.write(k + " , " + formatter.format((double) sumProfit / MAX_REPEAT) + " , ");
            bw.newLine();
        }
        bw.close();
        rep.close();
    }

    public static void chart_29_data_writer(double Profit) throws IOException {
        FileWriter rep = new FileWriter("q29_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
//        NumberFormat formatter = new DecimalFormat("#0.00");
        bw.write(Profit + " , ");
        bw.newLine();
        bw.close();
        rep.close();
    }

    public static void chart_30_data_writer(int temperature, double variance) throws IOException {
        FileWriter rep = new FileWriter("q30_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        NumberFormat formatter = new DecimalFormat("#0.00");
        bw.write(temperature + " , " + formatter.format(variance) + " , ");
        bw.newLine();
        bw.close();
        rep.close();
    }


    /**
     * Depending on your problem, generate random integers/doubles and store
     * them in the static lists you've defined in the SA_Algorithm class.
     * <p>
     * for example, if you have the Job Scheduling problem, you must generate
     * random data for a Job object which includes a list of profits, deadlines
     * and execution times.
     *
     * @param configSize the size of the list(s) which are going to be generated (value
     *                   of "n")
     */
    public static void generateRandomData(int configSize) {
        SA_Algorithm.knapsack_weight = (5 + (int) (Math.random() * 11)) * configSize;//random size between 5*n to 15*n
//        System.out.println(SA_Algorithm.knapsack_weight);
        SA_Algorithm.weights = new int[configSize];
        SA_Algorithm.values = new int[configSize];
        for (int i = 0; i < configSize; i++) {

            SA_Algorithm.weights[i] = 1 + (int) (Math.random() * 50);//random number between 1 to 50
            SA_Algorithm.values[i] = 1 + (int) (Math.random() * 100);//random number between 1 to 100

        }
    }
}
