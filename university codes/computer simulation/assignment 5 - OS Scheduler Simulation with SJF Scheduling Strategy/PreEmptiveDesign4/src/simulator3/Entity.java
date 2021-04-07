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
public class Entity {
    int type;
    int arrivalTime;
    int serviceTime;
    int preEmptedTime=-1;  
    public Entity(int t,int a,int s)
    {
        type=t;
        arrivalTime=a;
        serviceTime=s;
    }
    public void preEmpte(int currSerTime)
    {
        preEmptedTime=Executer.clock;
        serviceTime=currSerTime-Executer.clock; // remianing service time
    }
    public int waitingTime()
    {
        if (preEmptedTime>=0)
            return Executer.clock-preEmptedTime;
        else
            return Executer.clock-arrivalTime;
    }
    public int TurnAroundTime()
    {
        return Executer.clock-arrivalTime;
    }
    
    
}

