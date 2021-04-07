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
        System.out.println("chart 1");
        chart_1();
        System.out.println("chart 2");
        chart_2();
        System.out.println("chart 3");
        chart_3();
        System.out.println("chart 4");
        chart_4();
        System.out.println("chart 5");
        chart_5();
        System.out.println("chart 6");
        chart_6();
//        chart_7();
    }

//    public static void chart_23() throws IOException {
//        FileWriter rep = new FileWriter("q23_chart.txt", true);
//        BufferedWriter bw = new BufferedWriter(rep);
//        NumberFormat formatter = new DecimalFormat("#0.00");
//        bw.write("K , Average Profit,");
//        bw.newLine();
//        generateRandomData(50);
//        for (double k = 0.5; k <= MAX_K; k += 0.5) {
//            double sumProfit = 0;
//            SA_Algorithm.K = k;
//            for (int i = 0; i < MAX_REPEAT; i++) {
//                SA_Algorithm.start();
//                sumProfit += SA_Algorithm.profit;
//            }
//            bw.write(k + " , " + formatter.format((double) sumProfit / MAX_REPEAT) + " , ");
//            bw.newLine();
//        }
//        bw.close();
//        rep.close();
//    }

//    public static void chart_29_data_writer(double Profit) throws IOException {
//        FileWriter rep = new FileWriter("q29_chart.txt", true);
//        BufferedWriter bw = new BufferedWriter(rep);
////        NumberFormat formatter = new DecimalFormat("#0.00");
//        bw.write(Profit + " , ");
//        bw.newLine();
//        bw.close();
//        rep.close();
//    }

//    public static void chart_30_data_writer(int temperature, double variance) throws IOException {
//        FileWriter rep = new FileWriter("q30_chart.txt", true);
//        BufferedWriter bw = new BufferedWriter(rep);
//        NumberFormat formatter = new DecimalFormat("#0.00");
//        bw.write(temperature + " , " + formatter.format(variance) + " , ");
//        bw.newLine();
//        bw.close();
//        rep.close();
//    }


    /**
     * Depending on your problem, generate random integers/doubles and store
     * them in the static lists you've defined in the SA_Algorithm class.
     * <p>
     * for example, if you have the Job Scheduling problem, you must generate
     * random data for a Job object which includes a list of profits, deadlines
     * and execution times.
     * <p>
     * //     * @param configSize the size of the list(s) which are going to be generated (value
     * of "n")
     */

    public static void chart_1() throws IOException {
        FileWriter rep = new FileWriter("q1_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("cost coefficient 1(for number of Queues) , number of Queues, ");
        bw.newLine();
        NumberFormat formatter = new DecimalFormat("#0.00");
        for (double i = 100; i <= 2100; i += 200) {
            SA_Coordinator_Part_I.generateRandomData(1000000, i, 400, 3100, 2, 3, 9);
            SA_Algorithm.start();
//            System.out.println((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.write((int) i + " , " + SA_Algorithm.config[0] + " , ");
            bw.newLine();
        }
        bw.close();
        rep.close();
    }

    public static void chart_2() throws IOException {
        FileWriter rep = new FileWriter("q2_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("cost coefficient 2(for number of Queues) , number of CPUs , ");
        bw.newLine();
        NumberFormat formatter = new DecimalFormat("#0.00");
        for (double i = 200; i <= 2900; i += 200) {
            SA_Coordinator_Part_I.generateRandomData(1000000, 100, i, 3100, 2, 3, 9);
            SA_Algorithm.start();
//            System.out.println((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.write((int) i + " , " + SA_Algorithm.config[1] + " , ");
            bw.newLine();
        }
        bw.close();
        rep.close();
    }

    public static void chart_3() throws IOException {
        FileWriter rep = new FileWriter("q3_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("cost coefficient 1(for number of Queues) , Average Waiting Time , ");
        bw.newLine();
        NumberFormat formatter = new DecimalFormat("#0.00");
        for (double i = 100; i <= 2100; i += 200) {
            SA_Coordinator_Part_I.generateRandomData(1000000, i, 400, 3100, 2, 3, 9);
            SA_Algorithm.start();
//            System.out.println((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.write((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.newLine();
        }
        bw.close();
        rep.close();
    }

    public static void chart_4() throws IOException {
        FileWriter rep = new FileWriter("q4_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("cost coefficient 2(for number of CPUs) , Average Waiting Time , ");
        bw.newLine();
        NumberFormat formatter = new DecimalFormat("#0.00");
        for (double i = 200; i <= 2900; i += 200) {
            SA_Coordinator_Part_I.generateRandomData(1000000, 100, i, 3100, 2, 3, 9);
            SA_Algorithm.start();
//            System.out.println((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.write((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.newLine();
        }
        bw.close();
        rep.close();
    }

    public static void chart_5() throws IOException {
        FileWriter rep = new FileWriter("q5_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("Budget , number of Queues , ");
        bw.newLine();
        NumberFormat formatter = new DecimalFormat("#0.00");
        for (double i = 1000; i <= 5000; i += 250) {
            SA_Coordinator_Part_I.generateRandomData(1000000, 100, 400, i, 2, 3, 9);
            SA_Algorithm.start();
//            System.out.println((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.write((int) i + " , " + SA_Algorithm.config[0] + " , ");
            bw.newLine();
        }
        bw.close();
        rep.close();
    }

    public static void chart_6() throws IOException {
        FileWriter rep = new FileWriter("q6_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("Budget , number of CPUs , ");
        bw.newLine();
        NumberFormat formatter = new DecimalFormat("#0.00");
        for (double i = 1000; i <= 5000; i += 250) {
            SA_Coordinator_Part_I.generateRandomData(1000000, 100, 400, i, 2, 3, 9);
            SA_Algorithm.start();
//            System.out.println((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.write((int) i + " , " + SA_Algorithm.config[1] + " , ");
            bw.newLine();
        }
        bw.close();
        rep.close();
    }

    public static void chart_7() throws IOException {
        FileWriter rep = new FileWriter("q7_chart.txt", true);
        BufferedWriter bw = new BufferedWriter(rep);
        bw.write("Budget , Average Waiting Time , ");
        bw.newLine();
        NumberFormat formatter = new DecimalFormat("#0.00");
        for (double i = 1000; i <= 5000; i += 250) {
            SA_Coordinator_Part_I.generateRandomData(1000000, 100, 400, i, 2, 3, 9);
            SA_Algorithm.start();
//            System.out.println((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.write((int) i + " , " + formatter.format(SA_Algorithm.profit) + " , ");
            bw.newLine();
        }
        bw.close();
        rep.close();
    }

    public static void generateRandomData(int simulationTime, double cost_coefficient1, double cost_coefficient2, double max_budget, int lambda, int delta, int mio) {

        SA_Algorithm.cost_coefficient1 = cost_coefficient1;
        SA_Algorithm.cost_coefficient2 = cost_coefficient2;
        SA_Algorithm.max_budget = max_budget;
        SA_Algorithm.lambda = lambda;
        SA_Algorithm.delta = delta;
        SA_Algorithm.mio = mio;
        SA_Algorithm.simulationTime = simulationTime;

    }
}
