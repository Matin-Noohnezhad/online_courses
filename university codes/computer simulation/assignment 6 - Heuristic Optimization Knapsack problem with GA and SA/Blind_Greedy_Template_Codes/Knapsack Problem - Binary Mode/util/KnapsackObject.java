package util;
/**
 * a class for an object for knapsack 0/1 problem.
 * @author Holakou Rahmanian
 *
 */
public class KnapsackObject {
	/**
	 * the price of the object
	 */
	private double price;
	/**
	 * the weight of the object
	 */
	private double weight;
	/**
	 * the ratio {price / weight} - needed in Greedy algorithm
	 */
	private double pToWRatio;
	
	public KnapsackObject(double price, double weight) {
		super();
		this.price = price;
		this.weight = weight;
		pToWRatio = price/weight;
	}
	public double getPrice() {
		return price;
	}
	public double getWeight() {
		return weight;
	}
	public double getpToWRatio() {
		return pToWRatio;
	}
	
}
