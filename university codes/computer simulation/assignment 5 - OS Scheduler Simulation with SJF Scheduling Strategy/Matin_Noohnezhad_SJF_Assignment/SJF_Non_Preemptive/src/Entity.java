
public class Entity {
//    int type;
    int arrivalTime;
    int serviceTime;
//    int preEmptedTime = -1;

    public Entity( int a, int s) {
//        type = t;
        arrivalTime = a;
        serviceTime = s;
    }

    public int difTime() {
        return Executer.clock - arrivalTime;
    }
}

