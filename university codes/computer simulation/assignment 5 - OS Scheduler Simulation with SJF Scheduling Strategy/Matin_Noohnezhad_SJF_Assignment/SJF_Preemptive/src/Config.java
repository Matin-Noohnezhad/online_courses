
public class Config {
    int numOfQueues;
    int numOfServers;
//    int numOfEntities;
    Queue[] queues;
    Server[] servers;

    public Config(int nq, int ns) {
//        numOfEntities = ne;
        numOfQueues = nq;
        numOfServers = ns;
        queues = new Queue[numOfQueues];
        for (int i = 0; i < numOfQueues; i++)
            queues[i] = new Queue();
        servers = new Server[numOfServers];
        for (int i = 0; i < numOfServers; i++)
//            servers[i] = new Server(numOfEntities);
            servers[i] = new Server();
    }

}
