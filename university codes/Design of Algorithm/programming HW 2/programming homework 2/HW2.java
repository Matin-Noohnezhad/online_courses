import java.util.Arrays;
import java.util.Scanner;


public class HW2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int[] coins = {1, 2, 8, 9};
		int amount = 32;

		System.out.println("Amount = " + amount);
		System.out.println("Coins   = " + Arrays.toString(coins));
		System.out.println("Cashier Coinage = " + Arrays.toString(CoinChanging.solveCashier(coins, amount)) + " (It must be [1, 2, 0, 3])");
		System.out.println("Optimal   Coinage = " + Arrays.toString(CoinChanging.solveBlind(coins, amount)));

		pauseProg();
		
		Coordinator.compareAlgs();
		pauseProg();

		Coordinator.verifyCorrectness();
		pauseProg();

		Coordinator.evaluateCorrectness();

	}
	
	public static void pauseProg(){
		System.out.println("Press enter to continue...");
		Scanner keyboard = new Scanner(System.in);
		keyboard.nextLine();
	}


}
