package util;
/**
 * A class for holding the results of an algorithm applied to a knapsack 0/1 problem
 * @author Holakou Rahmanian
 *
 */
public class KnapsackResult {
	private double totalPrice;
	private char selection[];
	private int numOfGeneratedNodes;
	public KnapsackResult(double totalPrice, char[] selection,
			int numOfGeneratedNodes) {
		super();
		this.totalPrice = totalPrice;
		this.selection = selection;
		this.numOfGeneratedNodes = numOfGeneratedNodes;
	}
	public double getTotalPrice() {
		return totalPrice;
	}
	public char[] getselection() {
		return selection;
	}
	public int getNumOfGeneratedNodes() {
		return numOfGeneratedNodes;
	}
	
	@Override
	public String toString() {
		return " Total Profit = "+totalPrice+", Selection = "+new String(selection)
		+ ", #Generated Nodes = "+numOfGeneratedNodes;
	}
}
