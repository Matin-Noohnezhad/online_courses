import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

/**
 * This class is a class for storing information of a polinomial like 
 * {a_n * x^n + a_n-1 * x^n-1 +... + a_1 * x + a_0}. There are various 
 * methods for different purposes of polynomial.
 * @author Holakou Rahmanian
 *
 */
public class Polynomial {
	/**
	 * a variable for storing all expressions of the polynomial
	 */
	private ArrayList<Expression> exps;
	/**
	 * a variable for storing the degree of the polynomial
	 */
	private int degree;
	/**
	 * a variable indicating whether the expressions of the polynomial
	 * are sorted or not. (just for printing purposes)
	 */
	private boolean isSorted;
	
	public Polynomial() {
		degree = -1;
		isSorted = false;
		exps = new ArrayList<Expression>();
	}
	@Override
	public String toString() {
		String result = "P(x) = ";
		if(!isSorted){
			isSorted = true;
		}
		Collections.sort(exps, new Comparator<Expression>(){

			public int compare(Expression arg0, Expression arg1) {
				
				return arg1.getPower()-arg0.getPower();
			}
			
		});
		for(int i=0; i<exps.size(); i++){
			result += exps.get(i).toString();
			if(i<exps.size()-1)
				result += "+ "; 
		}
		return result;
	}
	/**
	 * add a new expression to the polynomial
	 * @param exp the added expresssion
	 */
	public void addExpresssion(Expression exp){
		// change the degree of the polynomial if neccesary
		if(degree == -1){// if this is the first expression
			degree = exp.getPower();
		}else{// set degree if the power of expression is larger than the current degree
			degree = exp.getPower()> degree? exp.getPower(): degree;
		}
		
		exps.add(exp);
		isSorted = false;
	}
	
	/**
	 * A function for computing the polynomial
	 *  {a_n * x^n + a_n-1 * x^n-1 +... + a_1 * x + a_0} for 
	 * a specific value of {x}
	 * @param x the input value of {x} in polynomial {a_n * x^n + a_n-1 * x^n-1 +... + a_1 * x + a_0}
	 * @return the value of {a_n * x^n + a_n-1 * x^n-1 +... + a_1 * x + a_0}
	 */
	public double computeValue(double x){
	
	
	
	
	}
	

	/**
	 * a function for finding the integral polynomial of current polynomial
	 * @return the integral polynomial
	 */
	public Polynomial findIntegralPolynomial(){
	
		
	
	
	
		
	
	}
	
	/**
	 * compute the area under the curve of polynomial from "begin" to "end" with 
	 * "Definite Integral" i.e. using integral formula for polynomial.
	 * @param begin the begining of interval
	 * @param end the end of the interval
	 * @return the area under the curve of polynomial
	 */
	public double computeArea_DefiniteIntegral(double begin, double end){
	
	
	}

	/**
	 * The Simpson formula.
	 * @param a the begining of sub-interval
	 * @param b the end of sub-interval
	 * @return the estimated area under the curve of this sub-interval
	 */
	public double simpsonFormula(double a, double b){
	
	}
	
	/**
	 * estimate the area under the curve of polynomial from "begin" to "end" with 
	 * Simpson method i.e. break the interval into several sub-intervals and estimate the 
	 * area under each sub-interval with Simpson formula
	 * @param begin the begining of interval
	 * @param end the end of the interval
	 * @param h the length of each sub-interval
	 * @return the estimated area under the curve of polynomial
	 */
	public double computeArea_Simpson(double begin, double end, double h){
	
	
		
	
	
		
	
		
	
	}
	
	/**
	 * estimate the area under the curve of polynomial from "begin" to "end" with 
	 * Monte Carlo method i.e. finding the ratio of total random points located 
	 * under the curve 
	 * @param begin the begining of interval
	 * @param end the end of interval
	 * @param n number of total random points generated
	 * @return the estimated area under the curve of polynomial
	 */
	public double computeArea_MonteCarlo(double begin, double end, int n){
	
	
		
	
	
	
	
	
	
		
	
		
	
	}
}
