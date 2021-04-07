import java.util.ArrayList;

class Simulator {

    private int ST;
    private int clock;
    private int[] tbl;
    private ArrayList<Integer> QS;
    private ArrayList<Integer> QP;
    private int SOS1;
    private int SOS2;
    private int SQP;
    private long SQS;
    private int PCS1;
    private int PCS2;
    private int PCQS;
    private int PCQP;
    private int NP;
    private int NPd;
    private int NS;
    private long SWTS;
    private int SWTP;
    private int MP;
    private int MS;
    private boolean S1;
    private boolean S2;

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
        SOS1 = 0; // sum of service 1
        SOS2 = 0; // sum of service 2
        SQP = 0; // sum professor queue
        SQS = 0; // sum student queue
        PCS1 = 0; // previous clock server 1
        PCS2 = 0;// previous clock server 2
        PCQS = 0;// previous clock student queue
        PCQP = 0;// previous clock professor queue
        NP = 0; // number of professor served
        NPd = 0; // number of professor discarded
        NS = 0; // number of student served
        SWTS = 0; // sum of waiting time for students
        SWTP = 0; // sum of waiting time for professors
        MP = 0; // number of professor entered
        MS = 0;// number of student entered
        S1 = false; // server 1
        S2 = false; // server 2
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
        if (!S1) {
            NP++;
            S1 = true;
            PCS1 = clock;
            tbl[2] = this.clock + srvP();
        } else if (!S2) {
            NP++;
            S2 = true;
            PCS2 = clock;
            tbl[3] = this.clock + srvP();
        } else if (QP.size() > 10) {
            NPd++;
        } else {
            SQP += (clock - PCQP) * QP.size();
            PCQP = clock;
            QP.add(clock);
        }
        tbl[0] = clock + arrP();
    }

    private void sArrive() {
        MS++;
        if (!S1) {
            NS++;
            S1 = true;
            PCS1 = clock;
            tbl[2] = clock + srvS();
        } else if (!S2) {
            NS++;
            S2 = true;
            PCS2 = clock;
            tbl[3] = clock + srvS();
        } else {
            SQS += (clock - PCQS) * QS.size();
            PCQS = clock;
            QS.add(clock);
        }
        tbl[1] = clock + arrS();
    }

    private void endS1() {

        if (QP.size() > 0) {
            tbl[2] = clock + srvP();
            NP++;
            SQP += (clock - PCQP) * QP.size();
            PCQP = clock;
            SWTP += (clock - QP.remove(0));
        } else if (QS.size() > 0) {
            tbl[2] = clock + srvS();
            NS++;
            SQS += (clock - PCQS) * QS.size();
            PCQS = clock;
            SWTS += (clock - QS.remove(0));
        } else {
            SOS1 += clock - PCS1;
            tbl[2] = ST + 1;
            S1 = false;
        }
    }

    private void endS2() {
        if (QP.size() > 0) {
            tbl[3] = clock + srvP();
            NP++;
            SQP += (clock - PCQP) * QP.size();
            PCQP = clock;
            SWTP += (clock - QP.remove(0));
        } else if (QS.size() > 0) {
            tbl[3] = clock + srvS();
            NS++;
            SQS += (clock - PCQS) * QS.size();
            PCQS = clock;
            SWTS += (clock - QS.remove(0));
        } else {
            SOS2 += clock - PCS2;
            tbl[3] = ST + 1;
            S2 = false;
        }
    }

    private void report() {
        System.out.println("Number of professors which discard: " + NPd + " , percentage: " + (double) (NPd) * 100 / (double) (MP) + " %");
        System.out.println("Number of professors which get services: " + NP + " , percentage: " + (double) (NP) * 100 / (double) (MP) + " %");
        System.out.println("Number of students which get services: " + NS + " , percentage: " + (double) (NS) * 100 / (double) (MS) + " %");
        System.out.println("Number of professors which enters the system: " + MP + " , Number of students which enters the system: " + MS);
        System.out.println("Throughput of professors: " + (double) (NP) / (double) (ST) + " , Throughput of students: " + (double) (NS) / (double) (ST));
        System.out.println("Throughput of whole system: " + (((double) (NP) / (double) (ST)) + ((double) (NS) / (double) (ST))));
        if (S1) {
            SOS1 += ST - PCS1;
        }
        if (S2) {
            SOS2 += ST - PCS2;
        }
        System.out.println("Utilization of server 1: " + (double) (SOS1) / (double) (ST) + " , Utilization of server 2: " + (double) (SOS2) / (double) (ST));
        SQS += (ST - PCQS) * QS.size();
        SQP += (ST - PCQP) * QP.size();
        System.out.println("Average size of professors queue: " + (double) SQP / ST);
        System.out.println("Average size of students queue: " + (double) SQS / ST);
        while (QP.size() > 0) {
            SWTP += ST - QP.remove(0);
        }
        while (QS.size() > 0) {
            SWTS += ST - QS.remove(0);
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

public class ParallelQueueParallelServer {

    public static void main(String[] args) {
        new Simulator().simulate();
    }
}
