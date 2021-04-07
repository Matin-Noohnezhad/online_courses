
public class Entity {
    //    int type;
    double arrivalTime;
    double serviceTime;
    double preEmptedTime = -1;

    public Entity(double a, double s) {
//        type = t;
        arrivalTime = a;
        serviceTime = s;
    }

    public void preEmpte(double currSerTime) {
        preEmptedTime = Executer.clock;
        serviceTime = currSerTime - Executer.clock; // remianing service time
    }

    public double waitingTime() {
        if (preEmptedTime >= 0)
            return Executer.clock - preEmptedTime;
        else
            return Executer.clock - arrivalTime;
    }

    public double turnAroundTime() {
        return Executer.clock - arrivalTime;
    }
}

