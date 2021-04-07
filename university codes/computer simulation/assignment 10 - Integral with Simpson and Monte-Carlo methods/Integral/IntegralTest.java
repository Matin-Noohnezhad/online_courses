import javax.swing.JOptionPane;


public class IntegralTest {

	
	public static void main(String[] args) {
			while (true) {
				int choice = Integer
						.parseInt(JOptionPane
								.showInputDialog("Enter a number between 1-8 (else to exit):"
										+ "\n1 - Testing function <computeValue> in class <Expression>"
										+ "\n2 - Testing function <findIntegralExpression> in class <Expression>"
										+ "\n3 - Testing function <computeValue> in class <Polynomial>"
										+ "\n4 - Testing function <findIntegralPolynomial> in class <Polynomial>"
										+ "\n5 - Testing function <computeArea_DefiniteIntegral> in class <Polynomial>"
										+ "\n6 - Testing function <simpsonFormula> in class <Polynomial>"
										+ "\n7 - Testing function <computeArea_Simpson> in class <Polynomial>"
										+ "\n8 - Testing function <computeArea_MonteCarlo> in class <Polynomial>"));
				Expression e;
				Polynomial p = new Polynomial();
				p.addExpresssion(new Expression(4, 3));
				p.addExpresssion(new Expression(1, 2));
				p.addExpresssion(new Expression(2, 1));
				p.addExpresssion(new Expression(1, 0));
				double begin = 1, end = 4;
				switch (choice) {
				case 1:
					System.out
							.println("#1 - Testing function <computeValue> in class <Expression>:");
					e = new Expression(0.25, 5);
					System.out.println("The value of expression {" + e
							+ "} at {x = 2} is: " + e.computeValue(2)
							+ "\n(**NOTE: the answer should be 8)");
					break;
				case 2:
					System.out
							.println("\n#2 - Testing function <findIntegralExpression> in class <Expression>:");
					e = new Expression(10, 4);
					System.out.println("The integral of expression {" + e
							+ "} is: " + e.findIntegralExpression()
							+ "\n(**NOTE: the answer should be 2 * x^5)");
					break;
				case 3:
					System.out
							.println("\n#3 - Testing function <computeValue> in class <Polynomial>:");
					System.out.println("The value of polynomial {" + p
							+ "} at {x = 2} is: " + p.computeValue(2)
							+ "\n(**NOTE: the answer should be 41)");
					break;
				case 4:
					System.out
							.println("\n#4 - Testing function <findIntegralPolynomial> in class <Polynomial>:");
					System.out.println("The integral of polynomial {" + p
							+ "} is: " + p.findIntegralPolynomial()
							+ "\n(**NOTE: the answer should be "
							+ "x^4 + 1/3 x^3 + x^2 + x)");
					break;
				case 5:
					System.out
							.println("\n#5 - Testing function <computeArea_DefiniteIntegral> in class <Polynomial>:");
					System.out.println("The area under the polynomial {" + p
							+ "} from " + begin + " to " + end + " is: "
							+ p.computeArea_DefiniteIntegral(begin, end)
							+ "\n(**NOTE: the answer should be 294)");
					break;
				case 6:
					double a = 2,
					b = 3;
					System.out
							.println("\n#6 - Testing function <simpsonFormula> in class <Polynomial>:");
					System.out.println("The Simpson formula on polynomial {"
							+ p + "} from " + a + " to " + b + " is: "
							+ p.simpsonFormula(a, b)
							+ "\n(**NOTE: the answer should be 77.333)");
					break;
				case 7:
					double h = 0.01;
					System.out
							.println("\n#7 - Testing function <computeArea_Simpson> in class <Polynomial>:");
					System.out
							.println("The area under the polynomial {"
									+ p
									+ "} from "
									+ begin
									+ " to "
									+ end
									+ " with h = "
									+ h
									+ ""
									+ " is: "
									+ p.computeArea_Simpson(begin, end, h)
									+ "\n(**NOTE: the answer should be extremely close to 294 e.g. 294.0000000000001)");
					break;
				case 8:
					int n = 10000;
					System.out
							.println("\n#8 - Testing function <computeArea_MonteCarlo> in class <Polynomial>:");
					System.out
							.println("The area under the polynomial {"
									+ p
									+ "} from "
									+ begin
									+ " to "
									+ end
									+ " with n = "
									+ n
									+ ""
									+ " is: "
									+ p.computeArea_MonteCarlo(begin, end, n)
									+ "\n(**NOTE: the answer should be close to 294 e.g. something between 288 to 300)");
					break;

				default:
					System.exit(0);
					break;
				}
			}			
		
	}

}
