package ga;

/**
 * A class for generating data for drawing (Mutation probability - Executation
 * time), (Crossover probability - Executation time) and (Population size -
 * Executation time) curves for n = 50, 500 and 5000
 * 
 * @author MCS3108
 * 
 */
public class GA_Coordinator_Part_I {

	// The maximum size of populations

	static final int MAX_POP_SIZE = 500;

	static final double MAX_MUTATION_PROBABILITY = 0.01;

	static final double MAX_CROSSOVER_PROBABILITY = 0.9;

	// total number of repeats in each configuration size and mutation/crossover
	// probability

	static final int MAX_REPEAT = 5;

	/**
	 * this variable should be true when you want to generate data for (
	 * Mutation probability - Executation time) curve, otherwise it should be
	 * false.
	 */

	public static boolean mutation = true;

	/**
	 * this variable should be true when you want to generate data for (
	 * Crossover probability - Executation time) curve, otherwise it should be
	 * false.
	 */

	public static boolean crossover = false;

	/**
	 * this variable should be true when you want to generate data for (
	 * population size - Executation time) curve, otherwise it should be false.
	 */

	public static boolean population = false;

	public static void main(String args[]) {

		// variables for computing the running time of the algorithm
		long time, tempTime;

		System.out
				.println("config, mutation/crossover probability/population size, time");

		// a loop in which the size of configuration is increasing
		for (int configSize = 50; configSize <= 5000; configSize *= 10) {

			if (mutation) {

				// a loop in which the size of the configuration is fixed
				// but the mutation probability is increasing
				for (double k = 0.001; k < MAX_MUTATION_PROBABILITY; k += 0.001) {

					time = 0;

					// a loop in which both configurarion size and the mutation
					// probability are
					// fixed, but the algorithm is repeated.

					for (int i = 0; i < MAX_REPEAT; i++) {

						GA_Algorithm.list = new int[configSize];
						generateRandomData(configSize);
						tempTime = System.currentTimeMillis();
						GA_Algorithm.MUTATION_PROBABILITY = k;
						GA_Algorithm.genetic_function();
						tempTime = System.currentTimeMillis() - tempTime;
						time += tempTime;
					}
					time /= MAX_REPEAT;
					System.out.println(configSize + "," + k + "," + time);
				}
			} else if (crossover) {

				// a loop in which the size of the configuration is fixed
				// but the crossover probability is increasing

				for (double j = 0.1; j < MAX_CROSSOVER_PROBABILITY; j += 0.1) {
					time = 0;

					// a loop in which both configurarion size and the crossover
					// probability are
					// fixed, but the algorithm is repeated.

					for (int i = 0; i < MAX_REPEAT; i++) {

						GA_Algorithm.list = new int[configSize];

						generateRandomData(configSize);
						tempTime = System.currentTimeMillis();
						GA_Algorithm.CROSSOVER_PROBABILITY = j;
						GA_Algorithm.genetic_function();
						tempTime = System.currentTimeMillis() - tempTime;
						time += tempTime;
					}
					time /= MAX_REPEAT;
					System.out.println(configSize + "," + j + "," + time);
				}
			}
			if (population) {

				// a loop in which the size of the configuration is fixed
				// but the population size is increasing

				int temp = 10;
				int counter = 0;
				for (int j = 20; j < MAX_POP_SIZE; j += temp) {
					counter++;
					time = 0;

					// a loop in which both configurarion size and the
					// population size
					// are fixed, but the algorithm is repeated.

					for (int i = 0; i < MAX_REPEAT; i++) {

						GA_Algorithm.list = new int[configSize];

						generateRandomData(configSize);
						tempTime = System.currentTimeMillis();
						GA_Algorithm.POPULATION_SIZE = j;
						GA_Algorithm.genetic_function();
						tempTime = System.currentTimeMillis() - tempTime;
						time += tempTime;
					}
					time /= MAX_REPEAT;
					System.out.println(configSize + "," + j + "," + time);
					if (counter == 9)
						temp = 100;
				}
			}
		}

	}

	/**
	 * Depending on your problem, generate random integers/doubles and store
	 * them in the static lists you've defined in the GA_Algorithm class.
	 * 
	 * for example, if you have the OBST problem, you must generate random data
	 * ,probabilitis and failures and store them in the corresponding lists.
	 * 
	 * @param config
	 */
	public static void generateRandomData(int configSize) {

		
		
		
		
		
		
		
		
		
		
		
		
		
		
	}

}
