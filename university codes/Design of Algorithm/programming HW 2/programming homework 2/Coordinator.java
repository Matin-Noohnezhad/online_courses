import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;

public class Coordinator {

	public static final int MAX_NUM_COINS = 10;
	public static final int NUM_CONFIG = 100;
	public static final int REPEAT = 100;

	/**
	 * A method to compare Greedy algorithm with some other algorithm which always generate optimal solution
	 */
	public static void compareAlgs(){
		ArrayList<Integer> coinSizeList = new ArrayList<Integer>();
		ArrayList<Long> blindList = new ArrayList<Long>();
		ArrayList<Long> cashierList = new ArrayList<Long>();		

		int[] coins;
		int amount;
		long t0=0,t1=0,t2=0, blindTime, cashierTime;

		for(int c = 2; c < MAX_NUM_COINS ; c++){
			System.out.print("\nRunning for C = " + c);
			blindTime = cashierTime = 0;

			for(int j = 1; j < NUM_CONFIG; j++){
				System.out.print(".");
				coins = generateCoins(c);
				amount = generateAmount(coins[c-1]);
				t0 = System.currentTimeMillis();
				for(int i=1; i<REPEAT; i++){
					CoinChanging.solveCashier(coins, amount);
				}

				t1 = System.currentTimeMillis();
				for(int i=1; i<REPEAT; i++){
					CoinChanging.solveBlind(coins, amount);
				}

				t2 = System.currentTimeMillis();

				blindTime += t2-t1;
				cashierTime += t1-t0;

			}
			coinSizeList.add(c);
			blindList.add( (blindTime));
			cashierList.add( (cashierTime));

		}


		System.out.println("\n-------------------------TIME VS SIZE-------------------------------\n");

		PrintWriter writer;
		try {
			writer = new PrintWriter("time_size_data.csv", "UTF-8");

			System.out.println("Size, Dynamic, Greedy");
			writer.println("Size, Dynamic, Greedy");

			for(int j=0; j<coinSizeList.size(); j++){
				System.out.println(coinSizeList.get(j) + ", " + blindList.get(j) + ", " + cashierList.get(j));
				writer.println(coinSizeList.get(j) + ", " + blindList.get(j) + ", " + cashierList.get(j));
			}

			writer.close();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}

	/**
	 * Comparing the Greedy solution with optimal solution for set of coins whose values are powers of two 
	 */
	public static void verifyCorrectness(){
		int[] coins;
		int diff;
		System.out.println("\n--------------------DIFFERENCE BETWEEN GREEDY AND OPTIMAL (POWER OF 2 COINS) ------------------------------------\n");

		for(int c=2; c<6; c++){
			coins = generatePower2Coins(c);
			System.out.println("***** Coins = "+Arrays.toString(coins)+ " *****");

			for(int amount = 1; amount < 3*coins[c-1] ; amount++){
				diff = CoinChanging.sumIntArray(CoinChanging.solveCashier(coins, amount)) - CoinChanging.sumIntArray(CoinChanging.solveBlind(coins, amount));

				System.out.println("Amount = " + amount + ", Difference = " + diff);
			}
		}

	}

	/**
	 * Comparing the Greedy solution with optimal solution for set of arbitrary coins  
	 */
	public static void evaluateCorrectness(){
		int[] coins;
		int diff;
		int max_coin;
		System.out.println("\n--------------------DIFFERENCE BETWEEN GREEDY AND OPTIMAL (ARBITRARY COINS) ------------------------------------\n");

		for(int c=2; c<7; c++){
			coins = generateCoins(c);
			max_coin = coins[c-1];
			

			for(int amount = 1; amount < 3*max_coin ; amount++){
				coins = generateCoins(c);
				System.out.println("\nCoins = " + Arrays.toString(coins));
				diff = CoinChanging.sumIntArray(CoinChanging.solveCashier(coins, amount)) - CoinChanging.sumIntArray(CoinChanging.solveBlind(coins, amount));
				System.out.println("Amount = " + amount + ", Difference = " + diff);
			}
		}


	}

	public static int generateAmount(int max_coin){
		double amount = 10 * Math.random() * ( ( double ) max_coin); 

		return (int) Math.floor(amount);
	}

	public static int[] generateCoins(int num_coins){
		int[] coins = new int[num_coins];
		coins[0] = 1;

		double increase;

		for(int i=1; i<num_coins; i++){
			increase = Math.random() + 1.5;

			coins[i] = (int) Math.ceil(increase * ((double)coins[i-1]));

		}

		return coins;
	}

	public static int[] generatePower2Coins(int num_coins){
		int[] coins = new int[num_coins];
		coins[0] = 1;

		for(int i=1; i<num_coins; i++){

			coins[i] = (int) Math.pow(2, i);

		}

		return coins;
	}
}
