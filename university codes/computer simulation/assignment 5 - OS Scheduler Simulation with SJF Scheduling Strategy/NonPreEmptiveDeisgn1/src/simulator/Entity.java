/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package simulator;

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
    public int difTime()
    {
        return Executer.clock-arrivalTime;
    }
}

