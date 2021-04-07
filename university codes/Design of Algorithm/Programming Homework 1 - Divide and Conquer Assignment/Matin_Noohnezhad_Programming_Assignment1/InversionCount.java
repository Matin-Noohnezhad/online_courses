package org.bihe;

import java.util.Arrays;


public class InversionCount {

    /**
     * Brute-Force algorithm for counting the number of inversions in a given array.
     *
     * @param a an array of integers
     * @return number of inversions
     */
    public static int bruteForce(int[] a) {
        /*
         * Use two nested loops that iterates over all pairs of "a"
         * and check whether each pair forms an inversion.
         */
        int numberOfInversion = 0;

        for (int i = 0; i < a.length; i++) {
            for (int j = i + 1; j < a.length; j++) {
                if (a[i] > a[j]) {
                    numberOfInversion++;
                }
            }
        }


        return 0;
    }

    /**
     * Divide and Conquer algorithm for counting the number of inversions in a given array.
     *
     * @param a an array of integers
     * @return number of inversions
     */
    public static int sortAndCount(int[] a) {

        if (a.length == 1) {
            return 0;
        } else {
            int[] firstHalf = new int[a.length / 2];
            int[] secondHalf = new int[a.length - a.length / 2];
            for (int i = 0; i < a.length; i++) {
                if (i < a.length / 2)
                    firstHalf[i] = a[i];
                else
                    secondHalf[i - a.length/2] = a[i];
            }
            int numberOfFirstHalfInversion = sortAndCount(firstHalf);
            int numberOfSecondHalfInversion = sortAndCount(secondHalf);
            int numberOfInterInversion = mergeAndCount(firstHalf, secondHalf, a);

            return numberOfFirstHalfInversion + numberOfSecondHalfInversion + numberOfInterInversion;

        }
    }

    /**
     * Merging to sorted list into a larger sorted list and count the number of inter-list inversions
     *
     * @param a_1 first sorted list
     * @param a_2 second sorted list
     * @param a   merged sorted list
     * @return number of inter-lists inversions
     */
    public static int mergeAndCount(int[] a_1, int[] a_2, int[] a) {

        int firstArrayCounter = 0;
        int secondArrayCounter = 0;
        int finalArrayCounter = 0;
        int numberOfInterInversion = 0;

        while (firstArrayCounter < a_1.length && secondArrayCounter < a_2.length) {
            if (a_1[firstArrayCounter] > a_2[secondArrayCounter]) {
                numberOfInterInversion += (a_1.length - firstArrayCounter);
                a[finalArrayCounter] = a_2[secondArrayCounter];
                finalArrayCounter++;
                secondArrayCounter++;
            } else {
                a[finalArrayCounter] = a_1[firstArrayCounter];
                finalArrayCounter++;
                firstArrayCounter++;
            }
        }
        if (firstArrayCounter == a_1.length) {
            while (finalArrayCounter < a.length) {
                a[finalArrayCounter] = a_2[secondArrayCounter];
                secondArrayCounter++;
                finalArrayCounter++;
            }
        } else if (secondArrayCounter == a_2.length) {
            while (finalArrayCounter < a.length) {
                a[finalArrayCounter] = a_2[firstArrayCounter];
                firstArrayCounter++;
                finalArrayCounter++;
            }
        }
        return numberOfInterInversion;
    }

}
