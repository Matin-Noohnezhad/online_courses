import java.io.IOException;

public class Main {

    /**
     * @param args
     */
    public static void main(String[] args) throws IOException {
        SA_Coordinator_Part_I.generateRandomData(1000000 , 100,400,3100, 2,3,9);
        SA_Algorithm.start();
    }

}
