/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package simulator3;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;


/**
 *
 * @author Hooman
 */
public class Observer 
{
    Config config;
    public int [] totalArrived; //0 professor 1 student
    public int [] totalServed;
    public int discarded;
    public long [] sumWaitingTime;
    public long [] sumTurnaroundTime; 
    public Observer(Config c)
    {
        config=c;// set the current configuration
        totalArrived=new int[c.numOfEntities]; // set variables for reporting
        totalServed=new int[c.numOfEntities];
        discarded=0;
        sumWaitingTime=new long[c.numOfEntities];
        sumTurnaroundTime=new long[c.numOfEntities];
        
    }
    public void entityArr(Entity e)
    {
        totalArrived[e.type-1]++;
    }
    public void entityExit(Entity e)
    {
        totalServed[e.type-1]++;
    }
    public void waitingTime(Entity e)
    {
        sumWaitingTime[e.type-1]+=e.waitingTime();
    }
    public void turnAroundTime(Entity e)
    {
        sumTurnaroundTime[e.type-1]+=e.TurnAroundTime();
    }
    public void report()
    {
        NumberFormat formatter = new DecimalFormat("#0.00");  
        
        for (int i=0;i<config.numOfQueues;i++) // Queues report
        {
            System.out.println( "Queue "+i+" Average: "+ formatter.format(config.queues[i].queueMean() ));
            System.out.println("Queue "+i+" Maximum: "+ config.queues[i].queueMax());
            System.out.println("Queue "+i+" Variance: "+ formatter.format(config.queues[i].queueVar()));
        }
        
        for (int i=0;i<config.numOfEntities;i++)//Entity report
        {
          System.out.println("Average waiting time for Entity "+ i+" without the ones in the queue: "+formatter.format((double)sumWaitingTime[i]/totalServed[i])); 
          System.out.println("Average Turnaround time for Entity "+ i+": "+formatter.format((double)sumTurnaroundTime[i]/totalServed[i])); // we dont calculate the last one in the server
          System.out.println( "Total Entity "+ i+" arrived: " +totalArrived[i] + " Total served:" +totalServed[i]+ " Served Percentage: "+formatter.format((double)totalServed[i]/totalArrived[i]*100)+" Throughput: "+formatter.format((double)totalServed[i]/Executer.simulationTime));
          while(!config.queues[i].isEmpty()) // adjust waiting time
            sumWaitingTime[i]+=(Executer.simulationTime-config.queues[i].remove().arrivalTime);
          System.out.println("Average adjusted waiting time for Entity "+ i+" "+formatter.format((double)sumWaitingTime[i]/totalArrived[i]));  
        }     
        System.out.println( "Total professors discarded: "+discarded +" Discarded Percentage: "+formatter.format((double)discarded/totalArrived[0]*100));
        
        for (int i=0;i<config.numOfServers;i++) // Servers report
        {
            double [] u= config.servers[i].utilization();
            for (int j=0;j<config.numOfEntities;j++)
            {
               System.out.println("Server "+ i+ " Entity "+j+" Utilization: "+ formatter.format(u[j])); 
            }
            System.out.println("Server "+i+" block time: "+formatter.format(config.servers[i].blockTime()));
            
        }
      
    }
    public void report2() throws IOException
    {
        FileWriter rep=new FileWriter("report.txt", true);
        BufferedWriter bw=new BufferedWriter(rep);
        NumberFormat formatter = new DecimalFormat("#0.00");  
        
         for (int i=0;i<config.parameters.length;i++)
         {
            bw.write(config.parameters[i]+",");
         }
        
        for (int i=0;i<config.numOfQueues;i++) // Queues report
        {
            bw.write(formatter.format(config.queues[i].queueMean() )+",");//"Queue "+i+" Average: "
            bw.write(config.queues[i].queueMax()+",");//"Queue "+i+" Maximum: "+
            bw.write(formatter.format(config.queues[i].queueVar())+",");//"Queue "+i+" Variance: "+ 
        }
        
        for (int i=0;i<config.numOfEntities;i++)//Entity report
        {
          bw.write(formatter.format((double)sumWaitingTime[i]/totalServed[i])+","); //"Average waiting time for Entity "+ i+" without the ones in the queue: "
          bw.write(formatter.format((double)sumTurnaroundTime[i]/totalServed[i])+","); // "Average Turnaround time for Entity "+ i+": "+
          bw.write( totalArrived[i] + ","+totalServed[i]+","+formatter.format((double)totalServed[i]/totalArrived[i]*100)+","+formatter.format((double)totalServed[i]/Executer.simulationTime)+",");
            //sumWaitingTime[i]+=(Executer.simulationTime-config.queues[i].remove().arrivalTime);
          //System.out.println("Average adjusted waiting time for Entity "+ i+" "+formatter.format((double)sumWaitingTime[i]/totalArrived[i]));  
        }     
        //System.out.println( "Total professors discarded: "+discarded +" Discarded Percentage: "+formatter.format((double)discarded/totalArrived[0]*100));
        
        for (int i=0;i<config.numOfServers;i++) // Servers report
        {
            double [] u= config.servers[i].utilization();
            for (int j=0;j<config.numOfEntities;j++)
            {
               bw.write(formatter.format(u[j])+","); 
            }
            //System.out.println("Server "+i+" block time: "+formatter.format(config.servers[i].blockTime()));
            
        }
        bw.newLine();
        bw.close();
        rep.close();
    }
    
}
