package org.bihe.helper;

import org.bihe.HashTable;
import org.bihe.probing.Probing;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Scanner;

public class HelperMethods {

    static Scanner sc = new Scanner(System.in);

    public static int sizeQuestioner() {
        boolean passTheQuestion = false;
        int htSize = 0;
        //
        while (!passTheQuestion) {
            System.out.print("Enter the size of a hash table.");
            try {
                htSize = sc.nextInt();
                passTheQuestion = true;
            } catch (Exception e) {
                System.err.println("Error: You must enter an integer which tells the size of your hash table.");
                System.out.println();
            }
        }
        //
        return htSize;
    }

    //
    public static int[] getNumbersFromUser(int noOfIntegers) {

        boolean passTheQuestion = false;
        int[] output = null;
        while (!passTheQuestion) {
            System.out.print("Enter " + noOfIntegers + " numbers in order, for inserting into you hash table.(The number should be in this format: n1, n2, n3, ...)");
            String input = sc.next();
            String[] arr = input.split(",");
            try {
                output = changeStringArray2IntArray(arr, noOfIntegers);
                passTheQuestion = true;
            } catch (Exception e) {
                System.err.println("You must enter " + noOfIntegers + " number of integers in this format: n1, n2, n3, ...");
            }
        }
        return output;
    }

    //
    public static int[] changeStringArray2IntArray(String[] arr, int noOfIntegers) throws Exception {
        if (arr.length != noOfIntegers) throw new Exception();
        int[] output = new int[noOfIntegers];
        for (int i = 0; i < noOfIntegers; i++) {
            output[i] = Integer.parseInt(arr[i].trim());
        }
        return output;
    }

    //
    public static void insertArrayInHashTable(HashTable hashTable, int[] arr, Probing probing) {
        for (int i = 0; i < arr.length; i++) {
            insertor(hashTable, arr[i], probing);
        }
    }

    //
    public static void insertor(HashTable hashTable, int input, Probing probing) {
        int i = 0;
        boolean insertStatus = false;
        while (true) {
            if (i > hashTable.getSize() - 1) {
                if (hashTable.allSlotsAreFull())
                    System.out.println("Your hash table is full !!!");
                else
                    System.out.println("Your hashing function having problem");
                break;
            }
            insertStatus = hashTable.insert(input, probing.hashingFunction(input, i));
            if (insertStatus) break;
            i++;
        }
    }

    //
    public static int[] generator(int arraySize) {
        int[] arr = new int[arraySize];
        for (int i = 0; i < arraySize; i++) {
            arr[i] = (int) (Math.random() * 100000);
        }
        return arr;
    }

    public static void writeInFile(String text) {
        PrintWriter writer = null;
        try {
            writer = new PrintWriter("ProbingTest.csv", "UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        writer.println(text);
        writer.close();
    }
}
