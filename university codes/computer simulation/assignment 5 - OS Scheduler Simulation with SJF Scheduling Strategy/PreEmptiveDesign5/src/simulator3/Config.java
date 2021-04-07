/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package simulator3;

/**
 *
 * @author Hooman
 */
public class Config 
{
    int numOfQueues;
    int numOfServers;
    int numOfEntities;
    Queue [] queues;
    Server [] servers;
     public int [] parameters;
    public Config( int ne,int nq,int ns, int [] p)
    {
        numOfEntities=ne;
        numOfQueues=nq;
        numOfServers=ns;
        parameters=p;
        queues=new Queue[numOfQueues]; 
        for (int i=0;i<numOfQueues;i++)
            queues[i]=new Queue();
        servers=new Server[numOfServers];
        for (int i=0;i<numOfServers;i++)
            servers[i]=new Server(numOfEntities);
    }
    
}
