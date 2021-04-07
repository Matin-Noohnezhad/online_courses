package org.bihe;

import java.util.Arrays;
import java.util.Scanner;


public class HW1 {

    public static void main(String[] args) {
        // TODO Auto-generated method stub

        int[] a = {1, 3, 2, 4, 1, -1};

        System.out.println("Testing org.bihe.InversionCount.bruteForce(.) ...");
        System.out.println("Sample array of integers: a = " + Arrays.toString(a));
        System.out.println("Number of inversions calculated by brute-force algorithm (it must be 9):");
        System.out.println("Your Answer ==> " + InversionCount.bruteForce(a));
        pauseProg();

        System.out.println("\n\nTesting org.bihe.InversionCount.mergeAndCount(.,.,.) ...");
        int[] A = {3, 7, 10, 14, 18};
        int[] B = {2, 11, 16, 17, 23, 50, 100};
        int[] C = {3, 7, 10, 14, 18, 2, 11, 16, 17, 23, 50, 100};
        System.out.println("Sample array of integers: ");
        System.out.println("\t A = " + Arrays.toString(A));
        System.out.println("\t B = " + Arrays.toString(B));
        System.out.println("Number of inter-lists inversions calculated by merge-and-count (it must be 9):");
        System.out.println("Your Answer ==> " + InversionCount.mergeAndCount(A, B, C));
        System.out.println("Merged sorted array must be C = [2, 3, 7, 10, 11, 14, 16, 17, 18, 23, 50, 100]");
        System.out.println("Your Answer     ======>     C = " + Arrays.toString(C));
        pauseProg();

        System.out.println("\n\nTesting org.bihe.InversionCount.sortAndCount(.) ...");
        System.out.println("Sample array of integers: a = " + Arrays.toString(a));
        System.out.println("Number of inversions calculated by brute-force algorithm (it must be 9):");
        System.out.println("Your Answer ==> " + InversionCount.sortAndCount(a));
        System.out.println("Merged sorted array must be a = [-1, 1, 1, 2, 3, 4]");
        System.out.println("Your Answer     ======>     a = " + Arrays.toString(a));
        pauseProg();

        Coordinator.compareAlgs();

    }

    public static void pauseProg() {
        System.out.println("Press enter to continue...");
        Scanner keyboard = new Scanner(System.in);
        keyboard.nextLine();
    }
}
