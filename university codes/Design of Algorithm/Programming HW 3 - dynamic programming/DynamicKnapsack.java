
public class DynamicKnapsack {
    private static int[][] M;
    private static int[] v;
    private static int[] w;

    /**
     * Solving the binary knapsack problem with dynamic programming through the iterative (or bottom-up) mode.
     *
     * @param v_in array of "values" of the objects
     * @param w_in array of "weights" of the objects
     * @param W_in the capacity of the knapsack
     * @return the maximum profit
     */
    public static int solveIterative(int[] v_in, int[] w_in, int W_in) {
        //Copy the inputs v_in and w_in into static variables v and w, respectively
        v = v_in;
        w = w_in;
        // "new" the matrix M
        M = new int[v.length + 1][W_in + 1];
        // Set the first row of M equal to zeros
        for (int i = 0; i <= W_in; i++) {
            M[0][i] = 0;
        }
        // Through two nested loops, go over the cells of M and fill in the values
        // based on the recurrence relation
        for (int i = 0; i <= W_in; i++) {
            for (int j = 1; j <= v.length; j++) {
                if (w_in[j - 1] > i)
                    M[j][i] = M[j - 1][i];
                else
                    M[j][i] = Math.max(M[j - 1][i], v_in[j - 1] + M[j - 1][i - w_in[j - 1]]);
            }
        }
        return M[v.length - 1][W_in];
    }

    /**
     * Solving the binary knapsack problem with dynamic programming through the memoized (or top-down) mode.
     *
     * @param v_in array of "values" of the objects
     * @param w_in array of "weights" of the objects
     * @param W_in the capacity of the knapsack
     * @return the maximum profit
     */
    public static int solveMemoized(int[] v_in, int[] w_in, int W_in) {
        //Copy the inputs v_in and w_in into static variables v and w, respectively
        v = v_in;
        w = w_in;
        // "new" the matrix M
        M = new int[v.length + 1][W_in + 1];

        // Initiate all entries of M into -1 indicating that the cell has not been processed
        for (int i = 0; i <= v.length; i++) {
            for (int j = 0; j <= W_in; j++) {
                M[i][j] = -1;
            }
        }

        // call the recursive function knapsackRecursive for values n and W_in
        return knapsackRecursive(v.length, W_in);
    }

    /**
     * A recursive function for finding the maximum profit of the knapsack problem
     *
     * @param num_of_obj
     * @param remaining_w
     * @return
     */
    private static int knapsackRecursive(int num_of_obj, int remaining_w) {
        // if the corresponding value of M is not -1, return the value
        if (M[num_of_obj][remaining_w] != -1)
            return M[num_of_obj][remaining_w];

        // if num_of_obj is zero, then then the answer is zero
        if (num_of_obj == 0 || remaining_w == 0)
            return 0;

        // otherwise go through the recurrence relation
        int value;
        if (remaining_w < w[num_of_obj - 1])
            value = knapsackRecursive(num_of_obj - 1,remaining_w);
        else
            value = Math.max(knapsackRecursive(num_of_obj - 1, remaining_w), v[num_of_obj - 1] + knapsackRecursive(num_of_obj - 1, remaining_w - w[num_of_obj - 1]));

        // store the value of the answer in the corresponding entry in M before returning it
        M[num_of_obj][remaining_w] = value;
        return value;
    }
}
