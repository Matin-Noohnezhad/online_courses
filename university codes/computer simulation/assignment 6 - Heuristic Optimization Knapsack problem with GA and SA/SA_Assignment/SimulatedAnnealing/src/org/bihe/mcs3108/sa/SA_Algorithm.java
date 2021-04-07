package org.bihe.mcs3108.sa;

/***
 * 
 * @author MCS3108
 * 
 */

public class SA_Algorithm {

	/**
	 * The initial temperature
	 */
	private static final int INITIAL_TEMPRETURE = 5000;
	
	/**
	 * The constant K
	 */
	public static double K = 0.5;

	/**
	 * The entry point of the program which contains the implementation of
	 * Simulated Annealing algorithm.
	 * 
	 * @param args
	 *            main method arguments
	 */
	public static void start() {

		int[] config = createRandomInitialConfig();
		double E = objectiveFunction(config);
		int T = INITIAL_TEMPRETURE;
		int unChangedCounter = 1;
		while (!terminate(T, unChangedCounter)) {
			for (int i = 0; i < counterDefiner(T); i++) {
				int[] config1 = changeConfig(config);
				double E1 = objectiveFunction(config1);
				if (isValid(config1)
						&& (E1 > E || (Math.random() < boltzmannFunction(E, E1,
								K, T)))) {
					config = config1;
					E = E1;
					unChangedCounter = 0;
				}
			}
			T = reduce(T);
			if (unChangedCounter != 0)
				unChangedCounter++;
		}

		printConfiguration(config);
		System.out.println();
		System.out.println("Result: "+ E);
	}
	
	/**
	 * Creates a random configuration
	 * 
	 * @return a random configuration
	 */
	private static int[] createRandomInitialConfig() {
	
	}

	/**
	 * Computes the objective value of a configuration
	 * 
	 * @param config
	 *            a configuration which represents a potential answer to the
	 *            problem
	 * @return the objective value of a configuration
	 */
	private static int objectiveFunction(int[] config) {
		
	}

	/***
	 * Implements the Boltzmann function
	 * 
	 * @param e
	 *            the best configuration result so far
	 * @param e1
	 *            the current temporary configuration result
	 * @param k
	 *            the temperature constant
	 * @param t
	 *            the current temperature
	 * @return the Boltzmann function result
	 */
	private static double boltzmannFunction(double e, double e1, double k, int t) {
		
	}

	/**
	 * Defines the number of times we change the configuration in a temperature
	 * 
	 * @param T
	 *            is the temperature
	 * @return
	 */
	private static int counterDefiner(int T) {
		
	}
	
	/**
	 * Change the configuration a little by some function
	 * 
	 * @param config
	 *            the current configuration
	 * @return the changed configuration
	 */
	private static int[] changeConfig(int[] config) {

	}
	
	/**
	 * Check to see if the current configuration is a valid configuration
	 * according to the problem definition
	 * 
	 * @param config
	 *            the current configuration
	 * @return true if configuration is valid and false otherwise
	 */
	private static boolean isValid(int[] config) {
		
	}
	
	/**
	 * @param T
	 *            current temperature
	 * @param unChangedCounter
	 *            number of times that the objective value did not changed
	 * @return boolean showing if algorithm should be terminated
	 */
	private static boolean terminate(int T, int unChangedCounter) {

	}

	/**
	 * Reduces the current temperature by some function
	 * 
	 * @param T
	 *            the current temperature
	 * @return the reduced temperature
	 */
	private static int reduce(int T) {
		int temp = T * 19 / 20;
		if (temp == T)
			return T--;
		else
			return temp;
	}

	private static void printConfiguration(int[] config) {
		String indexString = "";
		String configString = "";
		for (int i = 0; i < config.length; i++) {
			indexString += " " + i;
			configString += " " + config[i];
		}
		System.out.println("Index:         " + indexString);
		System.out.println("Configuration: " + configString);
	}
}
