package run;
import java.util.ArrayList;
import java.util.Random;

import util.KnapsackObject;

import algorithms.BlindAlgorithm;
import algorithms.GreedyAlgorithm;

/**
 * the coordinator class - a class for generating random configuration in order to test 
 * the algorithms and compare them together.
 * @author Holakou Rahmanian
 *
 */
public class Coordinator {
	// the maximum number of the  objects
	public static final int MAX_SIZE = 100000;
	
	// the number of different configuration which is supposed to be generated on each size
	public static final int NUM_OF_CONFIG = 10;
	
	// the number of repeats which is supposed to be done on each configuration
	public static final int NUM_OF_REPEAT = 3;
	
	public static void main(String[] args) {
		System.out.println("Size,\tBlind,\tGreedy" +
				"\n------------------------------------------------------------");
			
		// variables for storing time
		long time[] = new long[2];
		long startTime[] = new long[2];
		
		// capacity of the bag
		double m;
		
		ArrayList<KnapsackObject> objects;
		
		// a loop in which the number of objects is increased
		for(int size =1 ; size < MAX_SIZE ; size = increaseSize(size)){
			
			// a loop in which different configuration in one single size are generated
			for(int config = 1; config < NUM_OF_CONFIG ; config++){
				
				// generate new configuration
				m = createRandomCapacity(size);
				objects = createRandomObjects(size);
				
				startTime[0] = System.currentTimeMillis();
				
				// a loop in which the blind algorithm is applied repeatedly on the sample data
				for(int repeat = 1; repeat< NUM_OF_REPEAT ; repeat++){
					// apply algorithm to the data
					BlindAlgorithm.run((ArrayList<KnapsackObject>) objects.clone(), m);
				}
				time[0] += System.currentTimeMillis() - startTime[0];
				
				startTime[1] = System.currentTimeMillis();
				
				// a loop in which the greedy algorithm is applied repeatedly on the sample data
				for(int repeat = 1; repeat< NUM_OF_REPEAT ; repeat++){
					// apply algorithm to the data
					GreedyAlgorithm.run((ArrayList<KnapsackObject>) objects.clone(), m);
				}
				time[1] += System.currentTimeMillis() - startTime[1];
			}
			
			System.out.println(size+",\t"+(time[0]/(NUM_OF_CONFIG*NUM_OF_REPEAT))
					+",\t"+(time[1]/(NUM_OF_CONFIG*NUM_OF_REPEAT)));	
		}
		
	}
	
	/**
	 * Generate a random configuration (here: objects) of the knapsack 0/1 problem
	 * @param size the number of objects
	 * @return the objects
	 */
	public static ArrayList<KnapsackObject> createRandomObjects(int size) {
	/* Create "size" number of random knapsack objects. Use Gaussian distribution 
	*	(normal distribution) with mean 7 and standard deviation 4 for prices and use
	*	Gaussian distribution with mean 3 and standard deviation 2 for weights. 
	* 	Use an object from class Random and call nextGaussian() for Gaussian distribution.
	*	NOTE: it should NOT be negative; if it gets negative, use 0.1 instead
	*/


	







	


	


	

	}

	/**
	 * increase  the size of the problem in each step
	 * @param size the current size
	 * @return the new size
	 */
	public static int increaseSize(int size) {
	/* increase the current size by 10%; if it is too small, increase it by 1.
	*/


	}

	/**
	 * Generate a random configuration (here: the capacity) of the knapsack 0/1 problem
	 * @param size the number of objects
	 * @return the generated capacity
	 */
	public static double createRandomCapacity(int size){
	/* use Gaussian ditribution for generating the capacity of the knapsack
	*	with mean 2*size and standard deviation size/4
	* NOTE: it should NOT be negative; if it gets negative, use 0.1 instead
	*/


	}
}
