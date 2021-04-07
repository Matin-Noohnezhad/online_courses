import java.util.ArrayList;

class Simulator {

    private int ST;
    private int clock;
    private int[] tbl;
    private ArrayList<Integer> QS;
    private ArrayList<Integer> QP;
    //    private int SOS1;
    //    private int SOS2;
    private int SQP;
    private long SQS;
    private int PCS1;
    private int PCS2;
    private int PCQS;
    private int PCQP;
    private int NP;
    //    private int NPd;
    private int NS;
    private long SWTS;
    private int SWTP;
    private int MP;
    private int MS;
    private int S1;
    private int S2;
    //new variables(for preemptive system)
    private ArrayList<Integer> pESrv;
    private ArrayList<Integer> pETime;
    private int SOS1S;
    private int SOS1P;
    private int SOS2S;
    private int SOS2P;
    private int STAS;
    private int STAP;
    private int arrTime1;
    private int arrTime2;
    private int sum2S;
    private int sum2P;

    public Simulator() {
        ST = 1000000;//simulation time
        clock = 0;
        tbl = new int[4];
        tbl[0] = arrP();//professor arrival
        tbl[1] = arrS();//student arrival
        tbl[2] = ST + 1;//server 1 end of current service
        tbl[3] = ST + 1;//server 2 end of current service
        QS = new ArrayList<Integer>(); // queue of students
        QP = new ArrayList<Integer>();// queue of professors
//        SOS1 = 0; // sum of service 1
//        SOS2 = 0; // sum of service 2
        SQP = 0; // sum professor queue
        SQS = 0; // sum student queue
        PCS1 = 0; // previous clock server 1
        PCS2 = 0;// previous clock server 2
        PCQS = 0;// previous clock student queue
        PCQP = 0;// previous clock professor queue
        NP = 0; // number of professor served
//        NPd = 0; // number of professor discarded
        NS = 0; // number of student served
        SWTS = 0; // sum of waiting time for students
        SWTP = 0; // sum of waiting time for professors
        MP = 0; // number of professor entered
        MS = 0;// number of student entered
        S1 = 0; // server 1 (0 = empty , 1 = student , 2 = professor)
        S2 = 0; // server 2 (0 = empty , 1 = student , 2 = professor)
        //new variables(for preemptive system)
        pESrv = new ArrayList<Integer>(); // time service left for the students which preempt from servers
        pETime = new ArrayList<Integer>(); // time which preempt students, kicked out of servers
        SOS1S = 0; // sum of server 1 services for students
        SOS1P = 0; // sum of server 1 services for professors
        SOS2S = 0; // sum of server 2 services for students
        SOS2P = 0; // sum of server 2 services for professors
        arrTime1 = -1; // arrival time of somebody who getting service from server 1 right now. (for compute turn around time)
        arrTime2 = -1; // arrival time of somebody who getting service from server 1 right now. (for compute turn around time)
        sum2S = 0; // sum of Qi power of 2 for students queue
        sum2P = 0; // sum of Qi power of 2 for professors queue
    }

    public void simulate() {
        int i;
        while (true) {
            if (clock > ST) {
                report();
                break;
            } else {
                i = min(tbl);
                clock = tbl[i];
                switch (i) {
                    case 0:
                        pArrive(); // professor arrival event
                        break;
                    case 1:
                        sArrive();// student arrival event
                        break;
                    case 2:
                        endS1(); // end service of catering 1
                        break;
                    case 3:
                        endS2();// end service of catering 2
                        break;
                }
            }
        }
    }

    private void pArrive() {
        MP++;
        tbl[0] = clock + arrP();
        //
        if (QP.size() > 0 || (S1 == 2 && S2 == 2)) {
            SQP += (clock - PCQP) * QP.size();
            sum2P += Math.pow((clock - PCQP) * QP.size(), 2);
            PCQP = clock;
            QP.add(clock);
        } else if (S1 == 0) {
            S1 = 2;
            PCS1 = clock;
            tbl[2] = clock + srvP();
            arrTime1 = clock;
        } else if (S2 == 0) {
            S2 = 2;
            PCS2 = clock;
            tbl[3] = clock + srvP();
            arrTime2 = clock;
        } else if (S1 == 1) {
            SQS += (clock - PCQS) * (QS.size() + pETime.size());
            sum2S += Math.pow((clock - PCQS) * (QS.size() + pETime.size()), 2);
            PCQS = clock;
            pETime.add(clock);
            pESrv.add(tbl[2] - clock);
            SOS1S += clock - PCS1;
            PCS1 = clock;
            S1 = 2;
            tbl[2] = clock + srvP();
            STAS += clock - arrTime1;
            arrTime1 = clock;
        } else if (S2 == 1) {
            SQS += (clock - PCQS) * (QS.size() + pETime.size());
            sum2S += Math.pow((clock - PCQS) * (QS.size() + pETime.size()), 2);
            PCQS = clock;
            pETime.add(clock);
            pESrv.add(tbl[3] - clock);
            SOS2S += clock - PCS2;
            PCS2 = clock;
            S2 = 2;
            tbl[3] = clock + srvP();
            STAS += clock - arrTime2;
            arrTime2 = clock;
        }
        //
//        if (S1 == 0) {
////            NP++;
//            S1 = 1;
//            PCS1 = clock;
//            tbl[2] = this.clock + srvP();
//        } else if (S2 == 0) {
////            NP++;
//            S2 = 1;
//            PCS2 = clock;
//            tbl[3] = this.clock + srvP();
//        }
////        else if (QP.size() > 10) {
////            NPd++;
////        }
//        else {
//            SQP += (clock - PCQP) * QP.size();
//            PCQP = clock;
//            QP.add(clock);
//        }
    }

    private void sArrive() {
        MS++;
        tbl[1] = clock + arrS();
        if (S1 == 0) {
//            NS++;
            S1 = 1;
            PCS1 = clock;
            tbl[2] = clock + srvS();
            arrTime1 = clock;
        } else if (S2 == 0) {
//            NS++;
            S2 = 1;
            PCS2 = clock;
            tbl[3] = clock + srvS();
            arrTime2 = clock;
        } else {
            SQS += (clock - PCQS) * (QS.size() + pETime.size());
            sum2S += Math.pow((clock - PCQS) * (QS.size() + pETime.size()), 2);
            PCQS = clock;
            QS.add(clock);
        }
    }

    private void endS1() {

        //
        if (S1 == 1) {
            NS++;
            STAS += clock - arrTime1;
            SOS1S += clock - PCS1;
            PCS1 = clock;
        } else if (S1 == 2) {
            NP++;
            STAP += clock - arrTime1;
            SOS1P += clock - PCS1;
            PCS1 = clock;
        }
        if (QP.size() > 0) {
            //average of queue length
            SQP += (clock - PCQP) * QP.size();
            sum2P += Math.pow((clock - PCQP) * QP.size(), 2);
            PCQP = clock;
            //variance of queue length

            //
            arrTime1 = QP.remove(0);
            tbl[2] = clock + srvP();
            SWTP += clock - arrTime1;
            S1 = 2;
        } else if (pETime.size() > 0) {
            SQS += (clock - PCQS) * (QS.size() + pETime.size());
            sum2S += Math.pow((clock - PCQS) * (QS.size() + pETime.size()), 2);
            PCQS = clock;
            arrTime1 = pETime.remove(0);
            tbl[2] = clock + pESrv.remove(0);
            SWTS += (clock - arrTime1);
            S1 = 1;
        } else if (QS.size() > 0) {
            SQS += (clock - PCQS) * (QS.size() + pETime.size());
            sum2S += Math.pow((clock - PCQS) * (QS.size() + pETime.size()), 2);
            PCQS = clock;
            arrTime1 = QS.remove(0);
            tbl[2] = clock + srvS();
            SWTS += clock - arrTime1;
            S1 = 1;
        } else {
            S1 = 0;
            tbl[2] = ST + 1;
        }
        //
//        if (QP.size() > 0) {
//            tbl[2] = clock + srvP();
////            NP++;
//            SQP += (clock - PCQP) * QP.size();
//            PCQP = clock;
//            SWTP += (clock - QP.remove(0));
//        } else if (QS.size() > 0) {
//            tbl[2] = clock + srvS();
////            NS++;
//            SQS += (clock - PCQS) * QS.size();
//            PCQS = clock;
//            SWTS += (clock - QS.remove(0));
//        } else {
//            SOS1 += clock - PCS1;
//            tbl[2] = ST + 1;
//            S1 = 0;
//        }
    }

    private void endS2() {
        //
        if (S2 == 1) {
            NS++;
            STAS += clock - arrTime2;
            SOS2S += clock - PCS2;
            PCS2 = clock;
        } else if (S2 == 2) {
            NP++;
            STAP += clock - arrTime2;
            SOS2P += clock - PCS2;
            PCS2 = clock;
        }
        if (QP.size() > 0) {
            //average of queue length
            SQP += (clock - PCQP) * QP.size();
            sum2P += Math.pow((clock - PCQP) * QP.size(), 2);
            PCQP = clock;
            //variance of queue length

            //
            arrTime2 = QP.remove(0);
            tbl[3] = clock + srvP();
            SWTP += clock - arrTime2;
            S2 = 2;
        } else if (pETime.size() > 0) {
            SQS += (clock - PCQS) * (QS.size() + pETime.size());
            sum2S += Math.pow((clock - PCQS) * (QS.size() + pETime.size()), 2);
            PCQS = clock;
            arrTime2 = pETime.remove(0);
            tbl[3] = clock + pESrv.remove(0);
            SWTS += (clock - arrTime2);
            S2 = 1;
        } else if (QS.size() > 0) {
            SQS += (clock - PCQS) * (QS.size() + pETime.size());
            sum2S += Math.pow((clock - PCQS) * (QS.size() + pETime.size()), 2);
            PCQS = clock;
            arrTime2 = QS.remove(0);
            tbl[3] = clock + srvS();
            SWTS += clock - arrTime2;
            S2 = 1;
        } else {
            S2 = 0;
            tbl[3] = ST + 1;
        }
        //
//        if (QP.size() > 0) {
//            tbl[3] = clock + srvP();
////            NP++;
//            SQP += (clock - PCQP) * QP.size();
//            PCQP = clock;
//            SWTP += (clock - QP.remove(0));
//        } else if (QS.size() > 0) {
//            tbl[3] = clock + srvS();
////            NS++;
//            SQS += (clock - PCQS) * QS.size();
//            PCQS = clock;
//            SWTS += (clock - QS.remove(0));
//        } else {
//            SOS2 += clock - PCS2;
//            tbl[3] = ST + 1;
//            S2 = 0;
//        }
    }

    private void report() {
//        System.out.println("Number of professors which discard: " + NPd + " , percentage: " + (double) (NPd) * 100 / (double) (MP) + " %");
        System.out.println("Number of professors which get services: " + NP + " , percentage: " + (double) (NP) * 100 / (double) (MP) + " %");
        System.out.println("Number of students which get services: " + NS + " , percentage: " + (double) (NS) * 100 / (double) (MS) + " %");
        System.out.println("Number of professors which enters the system: " + MP + " , Number of students which enters the system: " + MS);
        System.out.println("Throughput of professors: " + (double) (NP) / (double) (ST) + " , Throughput of students: " + (double) (NS) / (double) (ST));
        System.out.println("Throughput of whole system: " + (((double) (NP) / (double) (ST)) + ((double) (NS) / (double) (ST))));
        if (S1 == 1) {
            SOS1S += clock - PCS1;
        } else if (S1 == 2) {
            SOS1P += clock - PCS1;
        }
        if (S2 == 1) {
            SOS2S += clock - PCS2;
        } else if (S2 == 2) {
            SOS2P += clock - PCS2;
        }
        System.out.println("Uitilization of server 1 for students: " + (double) SOS1S / ST);
        System.out.println("Uitilization of server 1 for professors: " + (double) SOS1P / ST);
        System.out.println("Uitilization of server 2 for students: " + (double) SOS2S / ST);
        System.out.println("Uitilization of server 2 for professors: " + (double) SOS2P / ST);
//        if (S1 == 1) {
//            SOS1 += ST - PCS1;
//        }
//        if (S2 == 1) {
//            SOS2 += ST - PCS2;
//        }
//        System.out.println("Utilization of server 1: " + (double) (SOS1) / (double) (ST) + " , Utilization of server 2: " + (double) (SOS2) / (double) (ST));
        SQS += (ST - PCQS) * (QS.size() + pETime.size());
        sum2S += Math.pow((ST - PCQS) * (QS.size() + pETime.size()), 2);
        SQP += (ST - PCQP) * QP.size();
        sum2P += Math.pow((ST - PCQP) * QP.size(), 2);
        System.out.println("Average size of professors queue: " + (double) SQP / ST);
        System.out.println("Average size of students queue: " + (double) SQS / ST);
        System.out.println("Variance of professors queue size: " + ((double) sum2P / ST - Math.pow((double) SQP / ST, 2)));
        System.out.println("Variance of students queue size: " + ((double) sum2S / ST - Math.pow((double) SQS / ST, 2)));
        System.out.println("Sum of Turn around time for professors: " + (double) STAP / NP);
        System.out.println("Sum of Turn around time for students : " + (double) STAS / NS);
        while (QP.size() > 0) {
            SWTP += ST - QP.remove(0);
        }
        while (QS.size() > 0) {
            SWTS += ST - QS.remove(0);
        }
        while (pETime.size() > 0) {
            pESrv.remove(0);
            SWTS += ST - pETime.remove(0);
        }
        System.out.println("Average waiting time for professors: " + (double) SWTP / MP);
        System.out.println("Average waiting time for students: " + (double) SWTS / MS);
    }

    private int min(int[] a) {
        int answer = 0;
        int min = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] < min) {
                min = a[i];
                answer = i;
            }
        }
        return answer;
    }

    private int arrP() {
        return 45 + (int) (Math.random() * 6);//45 to 50
    }//arrival time for professors

    private int arrS() {
        return 20 + (int) (Math.random() * 6);//20 to 25
    }//arrival time for students

    private int srvP() {
        return 40 + (int) (Math.random() * 6);//40 to 45
    }// service time for professors

    private int srvS() {
        return 35 + (int) (Math.random() * 6);//35 to 40
    }// service time for students

}

public class PreemptiveCateringSystem {

    public static void main(String[] args) {
        new Simulator().simulate();
    }
}

//enum Status {
//    EMP, STU, PROF
//}