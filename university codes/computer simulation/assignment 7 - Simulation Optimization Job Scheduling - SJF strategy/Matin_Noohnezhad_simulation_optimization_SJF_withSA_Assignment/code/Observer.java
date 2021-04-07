/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class Observer {
    //    public int[] totalArrived; //0 professor 1 student
    public int totalArrived; //0 professor 1 student
    //    public int[] totalServed;
    public int totalServed;
    //    public int discarded;
//    public long[] sumWaitingTime;
    public long sumWaitingTime;
    //    public long[] sumTurnaroundTime;
    public long sumTurnaroundTime;
    Config config;

    public Observer(Config c) {
        config = c;// set the current configuration
//        totalArrived = new int[c.numOfEntities]; // set variables for reporting
        totalArrived = 0; // set variables for reporting
//        totalServed = new int[c.numOfEntities];
        totalServed = 0;
//        discarded = 0;
//        sumWaitingTime = new long[c.numOfEntities];
        sumWaitingTime = 0;
//        sumTurnaroundTime = new long[c.numOfEntities];
        sumTurnaroundTime = 0;

    }

//    public void report() {
//        NumberFormat formatter = new DecimalFormat("#0.00");
//
//        for (int i = 0; i < config.numOfQueues; i++) // Queues report
//        {
//            System.out.println("Queue " + i + " Average: " + formatter.format(config.queues[i].queueMean()));
//            System.out.println("Queue " + i + " Maximum: " + config.queues[i].queueMax());
//            System.out.println("Queue " + i + " Variance: " + formatter.format(config.queues[i].queueVar()));
//        }
//
////        for (int i = 0; i < config.numOfEntities; i++)//Entity report
////        {
//        System.out.println("Average waiting time for jobs without the ones in the queue: " + formatter.format((double) sumWaitingTime / totalServed));
//        System.out.println("Average Turnaround time for jobs: " + formatter.format((double) sumTurnaroundTime / totalServed)); // we dont calculate the last one in the server
//        System.out.println("Total jobs arrived: " + totalArrived + " Total served:" + totalServed + " Served Percentage: " + formatter.format((double) totalServed / totalArrived * 100) + " Throughput: " + formatter.format((double) totalServed / Executer.simulationTime));
//        for (int i = 0; i < config.numOfQueues; i++) {
//            while (!config.queues[i].isEmpty()) // adjust waiting time
//                sumWaitingTime += (Executer.simulationTime - config.queues[i].remove().arrivalTime);
//        }
//        System.out.println("Average adjusted waiting time for jobs " + formatter.format((double) sumWaitingTime / totalArrived));
////        }
////        System.out.println("Total professors discarded: " + discarded + " Discarded Percentage: " + formatter.format((double) discarded / totalArrived[0] * 100));
//
//        for (int i = 0; i < config.numOfServers; i++) // Servers report
//        {
//            double u = config.servers[i].utilization();
////            for (int j = 0; j < config.numOfEntities; j++) {
//            System.out.println("CPU " + i + " jobs Utilization: " + formatter.format(u));
////            }
////            System.out.println("Server " + i + " block time: " + formatter.format(config.servers[i].blockTime()));
//
//        }
//
//    }

    public double report2() {
//        FileWriter rep = new FileWriter("report.txt", true);
//        BufferedWriter bw = new BufferedWriter(rep);
        NumberFormat formatter = new DecimalFormat("#0.00");
        for (int i = 0; i < config.numOfQueues; i++) {
            while (!config.queues[i].isEmpty()) // adjust waiting time
                sumWaitingTime += (Executer.simulationTime - config.queues[i].remove().arrivalTime);
        }
        //
//        bw.write(config.numOfServers + ",");
//        bw.write(config.numOfQueues + ",");
//        bw.write(Executer.mio + ",");
//        bw.write(Executer.delta + ",");
//        bw.write(Executer.lambda + ",");
//        bw.write(formatter.format((double) sumWaitingTime / totalServed) + ",");
        //
        return Double.parseDouble(formatter.format((double) sumWaitingTime / totalArrived));
//        bw.newLine();
//        bw.close();
//        rep.close();
    }

}
