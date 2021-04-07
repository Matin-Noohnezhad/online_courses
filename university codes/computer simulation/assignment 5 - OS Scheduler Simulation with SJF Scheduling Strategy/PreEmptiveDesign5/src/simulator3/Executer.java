
public class Executer {
    public static int simulationTime;
    public static int clock;
    //Queue pQueue=new Queue();
    //Queue sQueue=new Queue();
    //Server s1=new Server(2);
    //Server s2=new Server(2);
    public static Config config;
    //    public int [] totalArrived=new int[2]; //0 professor 1 student
//    public int [] totalServed=new int[2];
//    public int discardedP=0;
//    public int [] sumWaitingTime=new int[2];
//    public int [] sumTurnaroundTime=new int[2];
//    public int[] eventTable = new int[4];
    public int[] eventTable;
    Observer observer;
    public int queueRange = 4;

    public Executer(int simulationTime, int noQueues, int noCPUs) {
        config = new Config(noQueues, noCPUs);// 2 entities 2 queues 2 servers
        observer = new Observer(config);
        this.simulationTime = simulationTime;
        clock = 0;
        eventTable = new int[1 + noCPUs];
        eventTable[0] = jArr();
        for (int i = 1; i <= noCPUs; i++) {
            eventTable[i] = simulationTime + 1;
        }
//        eventTable[0] = pArr();
//        eventTable[1] = sArr();
//        eventTable[2] = simulationTime + 1;
//        eventTable[3] = simulationTime + 1;
    }

    public void simualte() {
        while (true) {
            int i = min(eventTable);
            clock = eventTable[i];
            if (clock > simulationTime) {
                report();
                break;
            }
            if (i == 0) {
                jArrival();
            } else {
                endService(i - 1);
            }
//            switch (i) {
//                case 0:
//                    pArrival();
//                    break;
//                case 1:
//                    sArrival();
//                    break;
//                case 2:
//                    endService1();
//                    break;
//                case 3:
//                    endService2();
//                    break;
//
//            }
        }
    }

    public void jArrival() {
        Entity e = new Entity(clock, jExec());//type=1 professor, arrival time =clock, service time =pSrv() we calculate it here
        observer.totalArrived++;
        eventTable[0] = clock + jArr();

        int situation = 0;
        //add to server
        for (int i = 0; i < config.numOfServers; i++) {
            if (config.servers[i].status == 0) {
                config.servers[i].occupy(e);
                eventTable[i + 1] = clock + e.serviceTime;
                situation = 1;
                break;
            }
        }

//        int queueRange = 4;
        int situation2 = 0;
        if (situation == 0) {
            if (e.serviceTime <= queueRange) {
                config.queues[0].add(e);
//                situation2 = 1;
            } else {
                for (int i = 1; i <= (config.numOfQueues - 2); i++) {
                    if (e.serviceTime <= (queueRange * (i + 1))) {
                        config.queues[i].add(e);
                        situation2 = 1;
                        break;
                    }
                }
                if (situation2 == 0) {
                    config.queues[config.numOfQueues - 1].add(e);
                }
            }
        }

//        if (config.servers[0].status() == 0) {
//            config.servers[0].occupy(e);
//            eventTable[2] = clock + e.serviceTime;
//        } else if (config.servers[1].status() == 0) {
//            config.servers[1].occupy(e);
//            eventTable[3] = clock + e.serviceTime;
//        }

//        else {
//            config.queues[0].add(e);
//        }


    }

//    public void sArrival() {
//        Entity e = new Entity(2, clock, sSrv());//type=2 student, arrival time =clock, service time =sSrv() we calculate it here
//        observer.totalArrived[1]++;
//        eventTable[1] = clock + sArr();
//        if (config.servers[0].status() == 0) {
//            config.servers[0].occupy(e);
//            eventTable[2] = clock + e.serviceTime;
//        } else if (config.servers[1].status() == 0) {
//            config.servers[1].occupy(e);
//            eventTable[3] = clock + e.serviceTime;
//        } else {
//            config.queues[1].add(e);
//        }
//    }

    public void endService(int serverNo) {
        Entity e = config.servers[serverNo].release();
        //crash if e in null because of an error
        observer.sumTurnaroundTime += (e.turnAroundTime()); //0 for professor and 1 for student
        observer.totalServed++;
        /////////////////////////////////////////////fix it later
        for (int i = 0; i < config.numOfQueues; i++) {
            if (!config.queues[i].isEmpty()) {
                e = config.queues[i].remove();
                observer.sumWaitingTime += e.waitingTime();
                config.servers[serverNo].occupy(e);
                eventTable[serverNo + 1] = clock + e.serviceTime;
                break;
            }
            if (i == config.numOfQueues - 1) {
                eventTable[serverNo + 1] = simulationTime + 1;
            }
        }


//        if (!config.queues[0].isEmpty()) {
//            e = config.queues[0].remove();
//            observer.sumWaitingTime += (e.difTime());
//            config.servers[serverNo].occupy(e);
//            eventTable[2] = clock + e.serviceTime;
//        } else if (!config.queues[1].isEmpty()) {
//            e = config.queues[1].remove();
//            observer.sumWaitingTime += (Executer.clock - e.arrivalTime);
//            config.servers[serverNo].occupy(e);
//            eventTable[2] = clock + e.serviceTime;
//        } else {
//            // we already released the server
//            eventTable[2] = simulationTime + 1;
//        }
        /////////////////////////////////////////////fix it later

    }

//    public void endService2() {
//        Entity e = config.servers[1].release();
//        //crash if e in null because of an error
//        observer.sumTurnaroundTime[e.type - 1] += (e.difTime()); //0 for professor and 1 for student
//        observer.totalServed[e.type - 1]++;
//        if (!config.queues[0].isEmpty()) {
//            e = config.queues[0].remove();
//            observer.sumWaitingTime[0] += (clock - e.arrivalTime);
//            config.servers[1].occupy(e);
//            eventTable[3] = clock + e.serviceTime;
//        } else if (!config.queues[1].isEmpty()) {
//            e = config.queues[1].remove();
//            observer.sumWaitingTime[1] += (clock - e.arrivalTime);
//            config.servers[1].occupy(e);
//            eventTable[3] = clock + e.serviceTime;
//        } else {
//            // we already released the server
//            eventTable[3] = simulationTime + 1;
//        }
//    }

    public void report() {
        observer.report();
    }

    private int min(int[] array) {
        int min = array[0];
        int index = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] < min) {
                min = array[i];
                index = i;
            }
        }
        return index;
    }

    //    private int pArr() {
//        return 1 + (int) (Math.random() * 1);
//    }
//
//    private int sArr() {
//        return 1 + (int) (Math.random() * 1);
//    }
//
//    private int pSrv() {
//        return 1 + (int) (Math.random() * 10);
//    }
//
//    private int sSrv() {
//        return 1 + (int) (Math.random() * 2);
//    }
    private int jArr() {
        float lambda = 2;
//        int a = (int) Math.ceil(-(1 / lambda) * Math.log(Math.random()));
//        System.out.println(a);
//        return a;
        return (int) Math.ceil(-(1 / lambda) * Math.log(Math.random()));
//        return 1 + (int) (Math.random() * 2);
    }

    private int jExec() {
        int delta = 3;
        int mio = 9;
        int n = 12;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += Math.random();
        }
        double standardNormalDistribution = (sum - n / 2) / Math.sqrt(n / 12);
        double nd = Math.ceil(3 * standardNormalDistribution + mio);
        if (nd < 1)
            return 1;
        return (int) nd;

//        return 1 + (int) (Math.random() * 2);
    }


}
