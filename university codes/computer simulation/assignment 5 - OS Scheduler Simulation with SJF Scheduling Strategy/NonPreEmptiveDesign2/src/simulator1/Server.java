/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package simulator1;

/**
 *
 * @author Hooman
 */
public class Server 
{
    public int status=0;  // 0 free, -1 block and any other number the type of the Entity which is being served
    int numEntities;
    public Entity srv=null;
    int [] sum; // Sum of Service for each entity
    int blockSum=0;
    int pClock=0; // when the current service is started
    public Server(int ne)
    {
        numEntities=ne;// how many different entities can be servered by this server 1 to n
        sum=new int[numEntities];    
    }
    public boolean occupy(Entity e)
    {
        if (status!=0)
            return false; // to be sure that the logic of simulation is correct    
        pClock=Executer.clock;
        srv=e; // put the entity in the server
        status=e.type; //type returns a number between 1 to n 
        return true;
    }
    public boolean block()
    {
        if (status<=0)
            return false; // block not possible already free or blocked
        sum[status-1]+=Executer.clock-pClock; // calculate the total service tiem for utilization
        status=-1;
        pClock=Executer.clock; // start time of blocking
        return true;
        
    }
    public Entity release()
    {
        if (status>0) //someone is being served
        {
            sum[status-1]+=Executer.clock-pClock; //calculate the total service tiem for utilization 
            status=0;
            return srv;
        }
        else if (status==-1) //server was blocked
        {
            blockSum+=Executer.clock-pClock;
            status=0;
            return srv;
        }
        else
            return null; // server was free
    }  
    public int status()
    {
        return status;  
    }
    public double [] utilization()
    {
        if (status>0) //last one in the server
            sum[status-1]+=Executer.simulationTime-pClock;
        else if ( status==-1)
            blockSum+=Executer.simulationTime-pClock;
   
        double [] u=new double [numEntities];
        for (int i=0;i<numEntities;i++)
        {
            u[i]= (double)sum[i]/Executer.simulationTime;
        }
        return u;
    }
    public double blockTime()
    {
            return (double)blockSum/Executer.simulationTime; 
    }
    
}
