
public class Entity {
    //    int type;
    int arrivalTime;
    int serviceTime;
    int preEmptedTime = -1;

    public Entity(int a, int s) {
//        type = t;
        arrivalTime = a;
        serviceTime = s;
    }

    public void preEmpte(int currSerTime) {
        preEmptedTime = Executer.clock;
        serviceTime = currSerTime - Executer.clock; // remianing service time
    }

    public int waitingTime() {
        if (preEmptedTime >= 0)
            return Executer.clock - preEmptedTime;
        else
            return Executer.clock - arrivalTime;
    }

    public int turnAroundTime() {
        return Executer.clock - arrivalTime;
    }
}

