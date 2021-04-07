/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package simulator2;

import java.text.DecimalFormat;
import java.text.NumberFormat;

/**
 *
 * @author Hooman
 */
public class Executer 
{
    public static int simulationTime;
    public static int clock;
//    public int [] totalArrived=new int[2]; //0 professor 1 student
//    public int [] totalServed=new int[2];
//    public int discardedP=0;
//    public int [] sumWaitingTime=new int[2];
//    public int [] sumTurnaroundTime=new int[2];
      public int [] eventTable =new int[4];
    //Queue pQueue=new Queue();
    //Queue sQueue=new Queue();
    //Server s1=new Server(2);
    //Server s2=new Server(2);
    public static Config config;
    Observer observer;
    public Executer(int st)
    {
        config=new Config(2,2,2);// 2 entities 2 queues 2 servers
        observer=new Observer(config);
        simulationTime=st;
        clock=0;
        eventTable[0]=pArr();
        eventTable[1]=sArr();
        eventTable[2]=simulationTime+1;
        eventTable[3]=simulationTime+1;
    }
    public void simualte()
    {
        while(true)
        {
            int i=min(eventTable);
            clock=eventTable[i];
            if (clock>simulationTime)
            {   
                report();
                break;
            }
            switch (i)
            {
                case 0:
                    pArrival();
                    break;
                case 1:
                    sArrival();
                    break;
                case 2:
                    endService1();
                    break;
                case 3:
                    endService2();
                    break;

            }
        }
            
    }
     public void pArrival()
        {
            Entity e=new Entity(1,clock,pSrv());//type=1 professor, arrival time =clock, service time =pSrv() we calculate it here
            observer.totalArrived[0]++;
            eventTable[0]=clock+pArr();
            if(config.queues[0].size()==0)
            {
                if (config.servers[0].status()==0)
                {
                    config.servers[0].occupy(e);
                    eventTable[2]=clock+e.serviceTime;
                    return;
                }
                else if ( config.servers[1].status()==0)
                {
                    config.servers[1].occupy(e);
                    eventTable[3]=clock+e.serviceTime; 
                    return;
                }
                else if ( config.servers[0].status()==2) // we are sure both serers are occupied
                {
                    Entity preEmpted=config.servers[0].release();
                    preEmpted.preEmptedTime=clock;
                    preEmpted.serviceTime=eventTable[2]-clock; // remianing serice time
                    config.queues[1].addPreEmpted(preEmpted);
                    config.servers[0].occupy(e);
                    eventTable[2]=clock+e.serviceTime;  
                    return;                    
                }
                else if ( config.servers[1].status()==2) // we are sure both serers are occupied
                {
                    Entity preEmpted=config.servers[1].release();
                    preEmpted.preEmptedTime=clock;
                    preEmpted.serviceTime=eventTable[3]-clock; // remianing serice time
                    config.queues[1].addPreEmpted(preEmpted);
                    config.servers[1].occupy(e);
                    eventTable[3]=clock+e.serviceTime; 
                    return;
                }         
            }
            if (config.queues[0].size()>10)
            {
                observer.discarded++;
            }
            else
            {
                config.queues[0].add(e);
            }      
        }
    public void sArrival()
        {
            Entity e=new Entity(2,clock,sSrv());//type=2 student, arrival time =clock, service time =sSrv() we calculate it here
            observer.totalArrived[1]++;
            eventTable[1]=clock+sArr();
            if ( config.servers[0].status()==0)
            {
                config.servers[0].occupy(e);
                eventTable[2]=clock+e.serviceTime;   
            }
            else if ( config.servers[1].status()==0)
            {
                config.servers[1].occupy(e);
                eventTable[3]=clock+e.serviceTime;   
            }
            else
            {
                config.queues[1].add(e);
            }      
        }
    public void endService1()
        {
            Entity e=config.servers[0].release();
            //crash if e in null because of an error
            observer.sumTurnaroundTime[e.type-1]+=(e.difTime()); //0 for professor and 1 for student
            observer.totalServed[e.type-1]++;
            if (!config.queues[0].isEmpty())
            {
                e= config.queues[0].remove();
                observer.sumWaitingTime[0]+=(e.difTime());
                config.servers[0].occupy(e);
                eventTable[2]=clock+e.serviceTime;
            }
            else if (!config.queues[1].isEmpty())
            {
                e= config.queues[1].remove();
                if (e.preEmptedTime>=0)
                    observer.sumWaitingTime[1]+=(clock-e.preEmptedTime);
                else
                    observer.sumWaitingTime[1]+=(e.difTime());
                config.servers[0].occupy(e);
                eventTable[2]=clock+e.serviceTime;
            }
            else
            {
                // we already released the server     
                eventTable[2]=simulationTime+1;
            }
                     
        }
    public void endService2()
        {
            Entity e=config.servers[1].release();
            //crash if e in null because of an error
          
            observer.sumTurnaroundTime[e.type-1]+=(e.difTime()); //0 for professor and 1 for student
            observer.totalServed[e.type-1]++;
            if (!config.queues[0].isEmpty())
            {
                e= config.queues[0].remove();
                observer.sumWaitingTime[0]+=(clock-e.arrivalTime); // or use difTime()
                config.servers[1].occupy(e);
                eventTable[3]=clock+e.serviceTime;
            }
            else if (!config.queues[1].isEmpty())
            {
                e= config.queues[1].remove();
                if (e.preEmptedTime>=0)
                    observer.sumWaitingTime[1]+=(clock-e.preEmptedTime);
                else
                    observer.sumWaitingTime[1]+=(e.difTime());
                config.servers[1].occupy(e);
                eventTable[3]=clock+e.serviceTime;
            }
            else
            {
                // we already released the server     
                eventTable[3]=simulationTime+1;
            }           
        }
    public void report()
    {
        
        observer.report();
    }
    private int min(int [] array)
        {
            int min=array[0];
            int index=0;
            for (int i=1;i<array.length;i++)
            {
                if (array[i]<min)
                {
                    min=array[i];
                    index=i;
                }
            }
            return index;
        }
        private int pArr()
        {
            return 1+(int)(Math.random()*1);
        }
        private int sArr()
        {
            return 1000+(int)(Math.random()*1);
        }
        private int pSrv()
        {
            return 1+(int)(Math.random()*10);
        }
        private int sSrv()
        {
            return 1+(int)(Math.random()*2);
        }
       
    
    
    
}
