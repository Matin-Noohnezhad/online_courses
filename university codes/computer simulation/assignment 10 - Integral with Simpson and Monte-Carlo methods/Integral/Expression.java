/**
 * This class is a class for storing information of an expression like {a * x^b}.
 * There are various methods for different purposes of expression.
 * @author Holakou Rahmanian
 *
 */
public class Expression {
	/**
	 * a variable for storing the parameter {a} of expression {a * x^b}
	 */
	private double coefficient;
	/**
	 * a variable fro storing the perameter {b} of expression {a * x^b}
	 */
	private int power;
	
	public Expression(double coefficient, int power) {
		super();
		this.coefficient = coefficient;
		this.power = power;
	}
	
	public double getCoefficient() {
		return coefficient;
	}
	public void setCoefficient(double coefficient) {
		this.coefficient = coefficient;
	}
	public int getPower() {
		return power;
	}
	public void setPower(int power) {
		this.power = power;
	}
	@Override
	public String toString() {
		if(power>0)
			return coefficient+" * x^"+power+" ";
		else
			return coefficient+" ";
	}
	
	/**
	 * A function for computing the expression {a * x^b} for a 
	 * specific value of {x}
	 * @param x the input value of {x} in expression {a * x^b} 
	 * @return the value of {a * x^b}
	 */
	public double computeValue(double x){
		
		
		
	
	
	}

	/**
	 * a function for finding the integral expression of current expression
	 * @return the integral expression or in other words the expression { (a/(b+1)) * x^(b+1)}
	 */
	public Expression findIntegralExpression(){
		
	}
}
