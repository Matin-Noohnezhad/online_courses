package ga;

import java.util.Random;
import java.util.Scanner;

/**
 * 
 * @author MCS3108
 * 
 */

public class GA_Algorithm {

	static Random random = new Random();

	public static int[] list;

	// This variable stores the population size
	public static int POPULATION_SIZE = 10;

	// This variable stores the mutation probability constant
	public static double MUTATION_PROBABILITY = 0.005;

	// This variable stores the mutation probability constant
	public static double CROSSOVER_PROBABILITY = 0.07;

	// This variable stores the number of populations
	public static int POPULATION_NUMBER = 10;

	// an integer variable that stores the number of created populations.
	// It will incremented by one when a new population is generated.
	static int counter = 1;

	// a double variable that stores the number of consecutive populations that
	// their
	// minimum cost has not improved.
	static double repeats = 0;

	// An array for storing the values of Roulette Weels.
	static double[] RWeels = new double[POPULATION_SIZE];;

	public static double genetic_function() {

		// get data from user
		getData();
		// Create initial population of chromosomes
		Population pop = Population.randomPopulation(list);
		// Sort the chromosomes of first population based on their cost.
		Population.sortByCost(pop.chromosomes);
		// Create Roulette Weels
		createRouletteWeels();
		// repeat following steps until stop criterion is satisfied.
		while (!terminated()) {

			// Generate next population
			Population nextPop = selection(pop);
			counter++;
			// Sort the chromosomes of new population based on their cost.
			Population.sortByCost(nextPop.chromosomes);
			// If the optimal cost has not improved increment "repeats" by one
			// otherwise, set its value to 0.
			if (nextPop.chromosomes.get(pop.chromosomes.size() - 1).cost >= pop.chromosomes
					.get(pop.chromosomes.size() - 1).cost)
				repeats++;
			else
				repeats = 0;

			for (int i = 0; i < pop.chromosomes.size(); i++) {
				pop.chromosomes.set(i, nextPop.chromosomes.get(i));
				pop.chromosomes.get(i).cost = nextPop.chromosomes.get(i).cost;
			}

		}

		// Output the best solution found
		return pop.chromosomes.get(pop.chromosomes.size() - 1).cost;
	}

	/**
	 * a function that selects chromosomes from the current population to be
	 * parents to crossover.
	 * 
	 * @param pop
	 *            (an object that refers to the current population)
	 * 
	 * @return the next population
	 */

	private static Population selection(Population pop) {

		Population nextPop = new Population();
		// An array that stores the two best configurations of current
		// population
		Tuple[] pr = selectTheTwoBest(pop);
		// Add the two best configurations to the next population
		nextPop.chromosomes.add(pr[0]);
		nextPop.chromosomes.add(pr[1]);

		// ***************************

		for (int k = 0; k < (GA_Algorithm.POPULATION_SIZE / 2) - 1; k++) {

			// Select two chromosomes from the previous population

			Tuple tpl1 = selectChromosome(pop);
			Tuple tpl2 = selectChromosome(pop);
			// check for crossover on selected chromosomes
			if (random.nextDouble() < GA_Algorithm.CROSSOVER_PROBABILITY) {

				Tuple[] pair = crossover(tpl1, tpl2);

				// check for mutation on each of selected chromosomes
				if (random.nextDouble() < GA_Algorithm.MUTATION_PROBABILITY) {
					Tuple tp = mutation(pair[0]);
					tp.cost = Tuple.getObjectiveFunction(tp);
					nextPop.chromosomes.add(tp);
				} else {
					Tuple tp = pair[0];
					tp.cost = Tuple.getObjectiveFunction(tp);
					nextPop.chromosomes.add(tp);
				}
				if (random.nextDouble() < GA_Algorithm.MUTATION_PROBABILITY) {
					Tuple tp = mutation(pair[1]);
					tp.cost = Tuple.getObjectiveFunction(tp);
					nextPop.chromosomes.add(tp);
				} else {
					Tuple tp = pair[1];
					tp.cost = Tuple.getObjectiveFunction(tp);
					nextPop.chromosomes.add(tp);

				}

			} else {
				// If no crossover was performed, add the selected chromosomes
				// to the next population.
				nextPop.chromosomes.add(tpl1);
				nextPop.chromosomes.add(tpl2);
			}
		}
		return nextPop;

	}

	/**
	 * A function for doing mutation.
	 * 
	 * @param tupple:
	 *            an object that refers to a tuple
	 * @return the tupple after mutation
	 */
	private static Tuple mutation(Tuple tupple) {





		


		



		
	}

	/**
	 * a function for doing crossover.
	 * 
	 * @param tpl1:
	 *            an object that refers to a tuple. It is one of the chromosomes
	 *            that have been selected in the selection function.
	 * @param tpl2:
	 *            an object that refers to a tuple. It is one of the chromosomes
	 *            that have been selected in the selection function.
	 * @return pair: an array that stores the two tuples after crossover.
	 * 
	 */
	private static Tuple[] crossover(Tuple tpl1, Tuple tpl2) {

		
		
		
		
		
		
		

		
		
		
		
		
		
		
		

		
		


		






		

	}

	/**
	 * This function selects a chromosome from current population using roulette
	 * weels selection
	 * 
	 * @param pop:
	 *            current population
	 * @return the selected chromosome
	 */
	private static Tuple selectChromosome(Population pop) {

		



		
		
		
		
		
		
		

	}

	/**
	 * A function for creating Roulette Weels. By using Roulette Weels
	 * selection, the better chromosomes have more chance to be selected. Store
	 * the values of Roulette Weels in the "RWeels" array
	 * 
	 */
	private static void createRouletteWeels() {

		
		



		

		

		
	}

	/**
	 * A function for selecting two best chromosomes from current population
	 * 
	 * @param pop :
	 *            current population
	 * @return an array that stores the two selected chromosomes
	 */
	private static Tuple[] selectTheTwoBest(Population pop) {

		
		
		

		
	}

	/**
	 * a function that specifies the end of genetic algorithm.
	 * 
	 * end criterion: if the number of created populations is equal with
	 * "POPULATION_NUMBER", or if for 50 consecutive populations, the objective
	 * function doesn't improved.
	 * 
	 * @return a boolean. If the end condition is satified the return value is
	 *         "true" otherwise, it's "false".
	 */

	private static boolean terminated() {

		
		
	}

	/**
	 * A function for getting data from user
	 * 
	 */
	public static void getData() {
		Scanner scan = new Scanner(System.in);
		System.out.println("Enter the number of nodes/jobs/cities: ");
		// variable for storing the number of nodes/jobs/cities
		int number = scan.nextInt();
		// provide the demension of array using the number
		list = new int[number];

		
		
		
		
		
		/**
		 * Complete the function
		 * 
		 */

		
		
		
	}
}
