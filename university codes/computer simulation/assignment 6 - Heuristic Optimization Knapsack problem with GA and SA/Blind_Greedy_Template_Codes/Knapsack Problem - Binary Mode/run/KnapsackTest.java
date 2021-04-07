package run;
import java.util.ArrayList;

import util.KnapsackObject;

import algorithms.BlindAlgorithm;
import algorithms.GreedyAlgorithm;

/**
 * Algorithms are tested here with sample data. 
 * @author Holakou Rahmanian
 *
 */
public class KnapsackTest {

	public static void main(String[] args) {
		
		// creating sample data for knapsack 0/1 problem
		ArrayList<KnapsackObject> objs = new ArrayList<KnapsackObject>();
		objs.add(new KnapsackObject(2, 1));
		objs.add(new KnapsackObject(6, 2));
		objs.add(new KnapsackObject(10,3));
		objs.add(new KnapsackObject(11,4));

		// running the algorithms
		System.out.println("Blind: "+BlindAlgorithm.run(objs, 7));
		System.out.println("Greedy: "+GreedyAlgorithm.run(objs, 7));
	}

}
