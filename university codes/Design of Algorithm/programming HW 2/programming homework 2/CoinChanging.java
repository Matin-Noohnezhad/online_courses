public class CoinChanging {

    public static int[][] opt_coinage = null;

    /**
     * Solving the coin changing problem using the greedy cashier algorithm
     *
     * @param coins  the set of coins
     * @param amount the amount of exchange
     * @return the number of coins used of each value
     */
    public static int[] solveCashier(int[] coins, int amount) {

        int[] answer = new int[coins.length];
        int maxPossibleIndex = coins.length - 1;
        int leftAmount = amount;

        while (leftAmount > 0) {
            if (coins[maxPossibleIndex] <= leftAmount) {
                answer[maxPossibleIndex]++;
                leftAmount -= coins[maxPossibleIndex];
            } else {
                if (maxPossibleIndex != 0)
                    maxPossibleIndex--;
            }
        }
        return answer;
    }


    /**
     * Solving the coin changing problem using the dynamic programming algorithm
     *
     * @param coins  the set of coins
     * @param amount the amount of exchange
     * @return the number of coins used of each value
     */
    public static int[] solveBlind(int[] coins, int amount) {
        int numCoins = coins.length;

        opt_coinage = new int[amount + 1][numCoins];

        for (int i = 0; i < opt_coinage.length; i++)
            opt_coinage[i] = null;

        return solveBlindRec(coins, amount);
    }

    /**
     * An auxiliary recursive function for solving the coin changing problem using the dynamic programming  algorithm
     *
     * @param coins  the set of coins
     * @param amount the amount of exchange
     * @return the number of coins used of each value
     */
    public static int[] solveBlindRec(int[] coins, int amount) {
        int numCoins = coins.length;

        // Return NULL if the amount is negative
        if (amount < 0)
            return null;

        // Return an array of zeros if the amount is zero
        if (amount == 0) {
            int[] coinage = new int[numCoins];
            for (int j = 0; j < numCoins; j++)
                coinage[j] = 0;
            return coinage;
        }

        // Check if the sub-problem has already been solved and the solution is stored
        if (opt_coinage[amount] != null) {
            return opt_coinage[amount].clone();
        }

        int[][] coinages = new int[numCoins][numCoins];

        // Try each coins and solve the corresponding sub-problem
        int[] opt_num_coins = new int[numCoins];
        for (int i = 0; i < numCoins; i++) {
            coinages[i] = solveBlindRec(coins, amount - coins[i]);
            opt_num_coins[i] = sumIntArray(coinages[i]);
        }

        // Find the best solution
        int min_index = 0;
        for (int j = 0; j < numCoins; j++)
            if (opt_num_coins[j] < opt_num_coins[min_index])
                min_index = j;

        int[] solution = coinages[min_index];
        if (solution != null)
            solution[min_index] += 1;

        // Store the solution to the array of solved problems
        opt_coinage[amount] = solution.clone();

        return solution;
    }


    public static int sumIntArray(int[] arr) {
        if (arr == null)
            return Integer.MAX_VALUE;

        int sum = 0;

        for (int a : arr)
            sum += a;

        return sum;
    }
}
