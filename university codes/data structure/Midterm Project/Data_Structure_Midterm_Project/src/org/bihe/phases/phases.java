package org.bihe.phases;

import org.bihe.HashTable;
import org.bihe.helper.HelperMethods;
import org.bihe.probing.LinearProbing;
import org.bihe.probing.QuadraticProbing;

public class phases {

    public static void phase1() {
        int htSize = HelperMethods.sizeQuestioner();
        HashTable ht = new HashTable(htSize);
        int noOfIntegers = htSize * 7 / 10;
        int[] arr = HelperMethods.getNumbersFromUser(noOfIntegers);
        //
        LinearProbing probing = new LinearProbing(htSize);
        HelperMethods.insertArrayInHashTable(ht, arr, probing);
        System.out.println("\n" + "hash table with linear probing.");
        ht.showTableContents();
        //
        HashTable ht2 = new HashTable(htSize);
        QuadraticProbing probing2 = new QuadraticProbing(htSize);
        HelperMethods.insertArrayInHashTable(ht2, arr, probing2);
        System.out.println("\n" + "hash table with quadratic probing.");
        ht2.showTableContents();
    }

    public static void phase2() {
        int repetition = 10;
        long startTime;
        long endTime;
        String forWriteInExcel = "size of table, mean of running time(linear)(nano second), mean of running time(quadratic)(nano second)\n";
        for (int i = 1000; i <= 20000; i += 500) {
            int htSize = i * 10 / 7;
            LinearProbing probing = new LinearProbing(htSize);
            QuadraticProbing probing2 = new QuadraticProbing(htSize);
            long sumLinearProbing = 0;
            long sumQuadraticProbing = 0;
            for (int j = 0; j < repetition; j++) {
                int[] datas = HelperMethods.generator(i);
                //linear probing
                HashTable ht = new HashTable(htSize);
                startTime = System.nanoTime();
                HelperMethods.insertArrayInHashTable(ht, datas, probing);
                endTime = System.nanoTime();
                sumLinearProbing += (endTime - startTime);
//                System.out.println("The running time of linear probing with " + i + " datas is: " + (endTime - startTime));
                //quadratic probing
                HashTable ht2 = new HashTable(htSize);
                startTime = System.nanoTime();
                HelperMethods.insertArrayInHashTable(ht2, datas, probing2);
                endTime = System.nanoTime();
                sumQuadraticProbing += (endTime - startTime);
//                System.out.println("The running time of quadratic probing with " + i + " datas is: " + (endTime - startTime));
            }
            //
            System.out.println("The mean of running time for table with size " + htSize + " with linear probing is: " + (int) (sumLinearProbing / repetition));
            System.out.println("The mean of running time for table with size " + htSize + " with quadratic probing is: " + (int) (sumQuadraticProbing / repetition));
            //
            forWriteInExcel = forWriteInExcel + htSize + ", " + (int) (sumLinearProbing / repetition) + " , " + (int) (sumQuadraticProbing / repetition) + "\n";
            //
        }
        HelperMethods.writeInFile(forWriteInExcel);

    }

    //---------------------------------------------


}
