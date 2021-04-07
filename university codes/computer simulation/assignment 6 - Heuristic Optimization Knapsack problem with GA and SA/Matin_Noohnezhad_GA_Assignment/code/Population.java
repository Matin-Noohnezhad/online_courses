import java.util.ArrayList;
import java.util.Random;

/**
 * This is a class for storing the information of a population. There are
 * various functions for different purposes in the Population class.
 *
 * @author MCS3108
 */
public class Population {

    // An array for storing the tuples
    ArrayList<Tuple> chromosomes = new ArrayList<Tuple>();

    static Random random = new Random();

    /**
     * @return the chromosomes
     */
    public ArrayList<Tuple> getChromosomes() {
        return chromosomes;
    }

    /**
     * @param chromosomes the chromosomes to set
     */
    public void setChromosomes(ArrayList<Tuple> chromosomes) {
        this.chromosomes = chromosomes;
    }

    /**
     * A function for generating a population from random tuples.
     *
     * @param list
     * @return the created population
     */
    public static Population randomPopulation(int[] list) {

        Population pop = new Population();
        ArrayList<Tuple> chromosomes = new ArrayList<Tuple>();

        for (int i = 0; i < GA_Algorithm.POPULATION_SIZE; i++) {
            chromosomes.add(Tuple.randomTuple(list));
        }
        pop.setChromosomes(chromosomes);
        return pop;
    }


    /**
     * A function for sorting the tuples based on their cost, decreasingly.
     *
     * @param lst an arrayList of tuples
     */

    public static void sortByCost(ArrayList<Tuple> lst) {
        int n = lst.size();
        for (int i = 1; i < n; i++) {
            Tuple m = lst.get(i);
            int j = i - 1;
            while ((j >= 0) && (lst.get(j).cost < m.cost))
                lst.set(j + 1, lst.get(j--));
            lst.set(j + 1, m);
        }
    }

}
