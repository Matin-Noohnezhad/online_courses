/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package simulator;

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
    public int [] totalArrived=new int[2]; //0 professor 1 student
    public int [] totalServed=new int[2];
    public int discardedP=0;
    public long [] sumWaitingTime=new long[2];
    public long [] sumTurnaroundTime=new long[2];
    public int [] eventTable =new int[4];
    Queue pQueue=new Queue();
    Queue sQueue=new Queue();
    Server s1=new Server(2);
    Server s2=new Server(2);
    public Executer(int st)
    {
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
            totalArrived[0]++;
            eventTable[0]=clock+pArr();
            if (s1.status()==0)
            {
                s1.occupy(e);
                eventTable[2]=clock+e.serviceTime;   
            }
            else if ( s2.status()==0)
            {
                s2.occupy(e);
                eventTable[3]=clock+e.serviceTime;   
            }
            else if (pQueue.size()>10)
            {
                discardedP++;
            }
            else
            {
                pQueue.add(e);
            }      
        }
    public void sArrival()
        {
            Entity e=new Entity(2,clock,sSrv());//type=2 student, arrival time =clock, service time =sSrv() we calculate it here
            totalArrived[1]++;
            eventTable[1]=clock+sArr();
            if ( s1.status()==0)
            {
                s1.occupy(e);
                eventTable[2]=clock+e.serviceTime;   
            }
            else if ( s2.status()==0)
            {
                s2.occupy(e);
                eventTable[3]=clock+e.serviceTime;   
            }
            else
            {
                sQueue.add(e);
            }      
        }
    public void endService1()
        {
            Entity e=s1.release();
            //crash if e in null because of an error
            sumTurnaroundTime[e.type-1]+=(clock-e.arrivalTime); //0 for professor and 1 for student
            totalServed[e.type-1]++;
            if (!pQueue.isEmpty())
            {
                e= pQueue.remove();
                sumWaitingTime[0]+=(clock-e.arrivalTime);
                s1.occupy(e);
                eventTable[2]=clock+e.serviceTime;
            }
            else if (!sQueue.isEmpty())
            {
                e= sQueue.remove();
                sumWaitingTime[1]+=(clock-e.arrivalTime);
                s1.occupy(e);
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
            Entity e=s2.release();
            //crash if e in null because of an error
            sumTurnaroundTime[e.type-1]+=(clock-e.arrivalTime); //0 for professor and 1 for student
            totalServed[e.type-1]++;
            if (!pQueue.isEmpty())
            {
                e= pQueue.remove();
                sumWaitingTime[0]+=(clock-e.arrivalTime);
                s2.occupy(e);
                eventTable[3]=clock+e.serviceTime;
            }
            else if (!sQueue.isEmpty())
            {
                e= sQueue.remove();
                sumWaitingTime[1]+=(clock-e.arrivalTime);
                s2.occupy(e);
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
        NumberFormat formatter = new DecimalFormat("#0.00");  
        System.out.println("Average waiting time for professors without the ones in the queue: "+formatter.format((double)sumWaitingTime[0]/totalServed[0]));
        System.out.println("Average waiting time for students without the ones in the queue: "+formatter.format((double)sumWaitingTime[1]/totalServed[1]));
        
        while(!pQueue.isEmpty())
            sumWaitingTime[0]+=(simulationTime-pQueue.remove().arrivalTime);
        while(!sQueue.isEmpty())
            sumWaitingTime[1]+=(simulationTime-sQueue.remove().arrivalTime);
        
        System.out.println("Average waiting time for professors: "+formatter.format((double)sumWaitingTime[0]/totalArrived[0]));
        System.out.println("Average waiting time for students : "+formatter.format((double)sumWaitingTime[1]/totalArrived[1]));
        
        System.out.println("Average Turnaround time for professors: "+formatter.format((double)sumTurnaroundTime[0]/totalServed[0])); // we dont calculate the last one in the server
        System.out.println("Average Turnaround time for students: "+formatter.format((double)sumTurnaroundTime[1]/totalServed[1]));// we dont calculate the last one in the server
        
        System.out.println( "Total professors arrived: " +totalArrived[0] + " Total professors served:" +totalServed[0]+ " Professor Served Percentage: "+formatter.format((double)totalServed[0]/totalArrived[0]*100)+" Professor Throughput: "+formatter.format((double)totalServed[0]/simulationTime));
        System.out.println( "Total professors discarded: "+discardedP +" Discarded Percentage "+formatter.format((double)discardedP/totalArrived[0]*100));
        System.out.println( "Total students arrived:" +totalArrived[1] + " Total students served:" +totalServed[1]+ " student Served Percentage "+formatter.format((double)totalServed[1]/totalArrived[1]*100)+" Student Throughput: "+formatter.format((double)totalServed[1]/simulationTime)); 
        
        double [] u= s1.utilization();
        System.out.println("Server 1 Professor Utilization: "+ formatter.format(u[0]));
        System.out.println("Server 1 Student Utilization: "+ formatter.format(u[1]));
        System.out.println("Server 1 block time: "+formatter.format(s1.blockTime()));
        
        u= s2.utilization();
        System.out.println("Server 2 Professor Utilization: "+ formatter.format(u[0]));
        System.out.println("Server 2 Student Utilization: "+ formatter.format(u[1]));
        System.out.println("Server 2 block time: "+formatter.format(s2.blockTime()));
        
        System.out.println("Professor Queue Average: "+ formatter.format(pQueue.queueMean() ));
        System.out.println("Professor Queue Maximum: "+ pQueue.queueMax());
        System.out.println("Professor Queue Variance: "+ formatter.format(pQueue.queueVar()));
        
        System.out.println("Student Queue Average: "+ formatter.format(sQueue.queueMean() ));
        System.out.println("Student Queue Maximum: "+ sQueue.queueMax());
        System.out.println("Student Queue Variance: "+ formatter.format(sQueue.queueVar() ));
          
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
            return 3+(int)(Math.random()*0);
        }
        private int sArr()
        {
            return 2+(int)(Math.random()*0);
        }
        private int pSrv()
        {
            return 5+(int)(Math.random()*0);
        }
        private int sSrv()
        {
            return 2+(int)(Math.random()*0);
        }
       
    
    
    
}
