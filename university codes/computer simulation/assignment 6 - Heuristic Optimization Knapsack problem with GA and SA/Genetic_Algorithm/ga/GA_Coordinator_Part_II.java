package ga;

/**
 * A class for generating data for drawing (Mutation probability - Average
 * optimal answer) , (Crossover probability - Average optimal answer) and (
 * Population size - Average optimal answer )curves for n = 50, 500 and 5000
 * 
 * @author MCS3108
 * 
 */
public class GA_Coordinator_Part_II {

	// The maximum size of populations

	static final int MAX_POP_SIZE = 500;

	static final double MAX_MUTATION_PROBABILITY = 0.01;

	static final double MAX_CROSSOVER_PROBABILITY = 0.9;

	// total number of repeats in each configuration size and mutation/crossover
	// probability

	static final int MAX_REPEAT = 5;

	/**
	 * this variable should be true when you want to generate data for (
	 * Mutation probability - optimal answer) curve, otherwise it should be
	 * false.
	 */

	static boolean mutation = true;

	/**
	 * this variable should be true when you want to generate data for (
	 * Crossover probability - optimal answer) curve, otherwise it should be
	 * false.
	 */

	static boolean crossover = false;

	/**
	 * this variable should be true when you want to generate data for (
	 * population size - optimal answer) curve, otherwise it should be false.
	 */

	public static boolean population = false;

	public static void main(String arg[]) {

		// variables for computing the optimal answer of the algorithm

		double opt_answer, average_opt_answer;

		System.out.println("config - mutation probability/crossover probability/Population size - optimal answer");

		// a loop in which the size of configuration is increasing

		for (int configSize = 50; configSize <= 5000; configSize *= 10) {
			GA_Algorithm.list = new int[configSize];
			GA_Coordinator_Part_I.generateRandomData(configSize);

			if (mutation) {

				// a loop in which the size of the configuration is fixed
				// but the mutation probability is increasing

				for (double k = 0.001; k < MAX_MUTATION_PROBABILITY; k += 0.001) {
					average_opt_answer = 0;
					GA_Algorithm.MUTATION_PROBABILITY = k;

					// a loop in which both configurarion size and the mutation
					// probability are
					// fixed, but the algorithm is repeated.

					for (int i = 0; i < MAX_REPEAT; i++) {
						opt_answer = GA_Algorithm.genetic_function();
						average_opt_answer += opt_answer;
					}
					average_opt_answer /= MAX_REPEAT;
					System.out.println(configSize + "," + k + ","
							+ average_opt_answer);
				}
			} else if (crossover) {

				// a loop in which the size of the configuration is fixed
				// but the crossover probability is increasing

				for (double j = 0.1; j < MAX_CROSSOVER_PROBABILITY; j += 0.1) {
					average_opt_answer = 0;
					GA_Algorithm.CROSSOVER_PROBABILITY = j;

					// a loop in which both configurarion size and the crossover
					// probability are
					// fixed, but the algorithm is repeated.

					for (int i = 0; i < MAX_REPEAT; i++) {
						opt_answer = GA_Algorithm.genetic_function();
						average_opt_answer += opt_answer;
					}
					average_opt_answer /= MAX_REPEAT;
					System.out.println(configSize + "," + j + ","
							+ average_opt_answer);
				}
			}
			if (population) {
				// a loop in which the size of the configuration is fixed
				// but the population size is increasing

				int temp = 10;
				int counter = 0;
				for (int j = 20; j < MAX_POP_SIZE; j += temp) {
					average_opt_answer = 0;
					counter++;

					// a loop in which both configurarion size and the
					// population size
					// are fixed, but the algorithm is repeated.

					for (int i = 0; i < MAX_REPEAT; i++) {
						opt_answer = GA_Algorithm.genetic_function();
						average_opt_answer += opt_answer;

					}
					average_opt_answer /= MAX_REPEAT;
					System.out.println(configSize + "," + j + ","
							+ average_opt_answer);
					if (counter == 9)
						temp = 100;
				}
			}
		}
	}
}
