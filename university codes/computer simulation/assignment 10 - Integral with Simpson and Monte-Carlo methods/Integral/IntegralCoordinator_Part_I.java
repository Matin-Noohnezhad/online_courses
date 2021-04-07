/**
 * A class for generating data for drawing first 24 curves
 * @author Holakou Rahmanian
 *
 */
public class IntegralCoordinator_Part_I {
	/**
	 * the maximum degree
	 */
	static final int MAX_DEGREE = 500;
	/**
	 * total number of configurations in each degree
	 */
	static final int MAX_CONFIG = 6;
	/**
	 * total number of repeats in each degree and configuration
	 */
	static final int MAX_REPEAT = 2;
	/**
	 * the parameter "n" in Monte-Carlo approach
	 */
	static final int n = 100;
	/**
	 * the parameter "h" in Simpson approach
	 */
	static final double h = 0.1;
	/**
	 * the beginning of interval on which the area is estimated
	 */
	static final int begin =0;
	/**
	 * the end of interval on which the area is estimated
	 */
	static final int end = 30;
	/**
	 * a boolean determining whether the appreach is Simpson or not (i.e. Monte-Carlo)
	 */
	static final boolean isSimpson = true;
	
	public static void main(String[] args) {
		System.out.println("degree,\ttime,\terror");
		
		// variables for computing the running time of the algorithm
		long time, tempTime;
		
		double error=0;
		Polynomial p;
		double estimatedArea, trueArea;
		
		// a loop in which the degree of the polynomial is increasing
		for(int degree = 1 ; degree<MAX_DEGREE; degree = increaseDegree(degree)){
			
			time =0;
			
			// a loop in which the degree of the polynomial is fixed
			// but the several configuration are generated
			for(int config =0; config<MAX_CONFIG; config ++){
				
				p = generateRandomPolynomial(degree);
				
				// a loop in which both degree and configuration of polynomial are
				// fixed, but the algorithm is repeated.
				for(int repeat=0 ; repeat<MAX_REPEAT; repeat++){
					
					tempTime =System.currentTimeMillis();
					if(isSimpson)
						estimatedArea = p.computeArea_Simpson(begin, end, h);
					else
						estimatedArea = p.computeArea_MonteCarlo(begin, end, n);
					tempTime =System.currentTimeMillis() - tempTime ;
					
					time += tempTime;
					
					// computing the error of this algorithm
					trueArea = p.computeArea_DefiniteIntegral(begin, end);
					error +=  (trueArea- estimatedArea)/trueArea;
				}
			}
			
			time /= MAX_CONFIG*MAX_REPEAT;
			error /= MAX_CONFIG*MAX_REPEAT;
			System.out.println(degree+",\t"+time+",\t"+error);
		}
		
	}

	/**
	 * a function which generate a random polynomial with given degree
	 * @param degree the degree of random polynomial
	 * @return a random polynomial
	 */
	public static Polynomial generateRandomPolynomial(int degree){
		Polynomial result = new Polynomial();
		for(int i=degree; i>=0; i--){
			result.addExpresssion(new Expression(10*Math.random(),i));
		}
		return result;
	}
	
	/**
	 * increase the degree of polynomial based on given old degree
	 * @param oldDegree the old degree
	 * @return the new degree
	 */
	public static int increaseDegree(int oldDegree){
		int newDegree = oldDegree*11/10;
		return newDegree>oldDegree ? newDegree: oldDegree+1;
	}
}
