package algorithms;

import java.util.ArrayList;

import util.KnapsackObject;
import util.KnapsackResult;
import util.KnapsackUtility;

/**
 * A class for developing Blind algorithm to solve Knapsack 0/1 problem.
 * @author Holakou Rahmanian
 *
 */
public class BlindAlgorithm {

	/**
	 * the blind algorithm
	 * @param objects the objects in knapsack problem
	 * @param m the capacity of the bag
	 */
	public static KnapsackResult run(ArrayList<KnapsackObject> objects, double m){
	/* 
	*	Iterate over all possible subsets i.e. all possible selections of objects while you are holding
	* 	the best answer so far. You can do it in a loop by "findNextSubset". For each subset (selection),
	*	check the feasibility (by "isFeasible") of it and check if its total profit (by "KnapsackUtility.profitOfSelection")
	*	is better than the best profit you record so far. If it is, store both its profit and selection. 
	* 	Remember to return an object of KnapsackResult as a result.
	*/
	


	


	


	



	


	


	


	



	
	
	
	
	



	



	

	}

	/**
	 * In the process of generating all the subsets, find the next subset based on the current one
	 * @param bitString the current selection/subset
	 * @param size the size of original set
	 * @return true if there still some subsets remaining
	 */
	public static boolean findNextSubset ( char bitString [], int size ) {
	/* in this method, implement the "findNextSubset" you see
	*	in the PDF of week 5 about blind algorithm
	*/


	






	




	}

	/**
	 * checking if the selected objects can fit in the bag
	 * @param objects all the objects
	 * @param m the capacity
	 * @param bitString the selection of objects 
	 * @return feasibility
	 */
	public static boolean isFeasible(ArrayList<KnapsackObject> objects, double m, char[] bitString){

	}
	
	
}
