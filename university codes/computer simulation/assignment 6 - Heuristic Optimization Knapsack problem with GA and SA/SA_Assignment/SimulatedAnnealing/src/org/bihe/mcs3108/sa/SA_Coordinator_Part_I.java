package org.bihe.mcs3108.sa;

public class SA_Coordinator_Part_I {

	/**
	 * The maximum value for K constant
	 */
	static final int MAX_K = 5;

	/**
	 * The number of times we repeat the algorithm for a same configuration
	 */
	static final int MAX_REPEAT = 5;

	public static void main(String args[]) {

		/**
		 * Variables to compute time
		 */
		long time, tempTime;

		System.out.println("config, K constant, time");

		// A loop in which the size of configuration is increasing
		for (int configSize = 50; configSize <= 5000; configSize *= 10) {

			// A loop in which the size of the configuration is fixed
			// but the value of k is increasing
			for (double k = 0.5; k < MAX_K; k += 0.5) {
				time = 0;

				// A loop in which both configuration and K are
				// fixed, but the algorithm is repeating
				for (int i = 0; i < MAX_REPEAT; i++) {

					// Your algorithm list(s) will be filled here
					// SA_Algorithm.list = new int[configSize];
					generateRandomData(configSize);
					tempTime = System.currentTimeMillis();
					SA_Algorithm.K = k;
					SA_Algorithm.start();
					tempTime = System.currentTimeMillis() - tempTime;
					time += tempTime;
				}
				time /= MAX_REPEAT;
				System.out.println(configSize + "," + k + "," + time);
			}
		}
	}

	/**
	 * Depending on your problem, generate random integers/doubles and store
	 * them in the static lists you've defined in the SA_Algorithm class.
	 * 
	 * for example, if you have the Job Scheduling problem, you must generate
	 * random data for a Job object which includes a list of profits, deadlines
	 * and execution times.
	 * 
	 * @param configSize
	 *            the size of the list(s) which are going to be generated (value
	 *            of "n")
	 */
	public static void generateRandomData(int configSize) {

	}
}
