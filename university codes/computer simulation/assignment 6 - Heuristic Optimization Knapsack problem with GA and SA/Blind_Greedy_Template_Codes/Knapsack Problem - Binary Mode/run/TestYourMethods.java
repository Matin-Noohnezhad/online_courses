package run;

import java.util.ArrayList;

import algorithms.BlindAlgorithm;
import algorithms.GreedyAlgorithm;

import util.KnapsackObject;
import util.KnapsackUtility;

/**
 * This class is written for testing your methods. Fill in each method and then
 * test what you wrote here.
 * @author Holakou Rahmanian
 *
 */
public class TestYourMethods {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		// creating sample data for knapsack 0/1 problem
		ArrayList<KnapsackObject> objs = new ArrayList<KnapsackObject>();
		objs.add(new KnapsackObject(2, 1));
		objs.add(new KnapsackObject(6, 2));
		objs.add(new KnapsackObject(10,3));
		objs.add(new KnapsackObject(11,4));

		// a sample selection
		char bitString[] = new char[4];
		bitString[0]='0';bitString[1]='1';bitString[2]='1';bitString[3]='0';

		// 1. uncomment to test KnapsackUtility.profitOfSelection -- Answer = 16.0		
		//System.out.println(KnapsackUtility.profitOfSelection(objs, bitString));

		// 2. uncomment to test KnapsackUtility.weightOfSelection -- Answer = 5.0
		//System.out.println(KnapsackUtility.weightOfSelection(objs, bitString));

		// 3. uncomment to test BlindAlgorithm.isFeasible -- Answer = false, true 
		//System.out.println(BlindAlgorithm.isFeasible(objs, 4, bitString)
		//		+", "+BlindAlgorithm.isFeasible(objs, 7, bitString));

		bitString[0]='1';bitString[1]='0';bitString[2]='1';bitString[3]='1';
		
		// 4. uncomment to test BlindAlgorithm.findNextSubset -- Answer = true, 1100
		//System.out.print(BlindAlgorithm.findNextSubset(bitString, 4));
		//System.out.println(", "+new String(bitString));
		
		// 5. uncomment to test BlindAlgorithm.run 
		//-- Answer =  Total Profit = 21.0, Selection = 0011, #Generated Nodes = 16
		//System.out.println(BlindAlgorithm.run(objs, 7));
		
		// 6. uncomment to test GreedyAlgorithm.run 
		//-- Answer = Total Profit = 18.0, Selection = 1101, #Generated Nodes = 1
		//System.out.println(GreedyAlgorithm.run(objs, 7));
		
		// 7. uncomment to test Coordinator.increaseSize -- Answer - 22, 5 
		//System.out.println(Coordinator.increaseSize(20)+", "+Coordinator.increaseSize(4));
		
		// 8 & 9. test Coordinator.createRandomObjects and Coordinator.createRandomCapacity yourself
		// note that since it is working with random data it's hard to check it.
	}

}
