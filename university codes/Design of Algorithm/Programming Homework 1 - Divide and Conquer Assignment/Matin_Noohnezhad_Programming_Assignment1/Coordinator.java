package org.bihe;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

public class Coordinator {
	
	public static final int MAX_N = 50000; 
	public static final int MIN_N = 1000; 
	public static final int REPEAT = 10;
	
	public static void compareAlgs(){
		ArrayList<Integer> sizeList = new ArrayList<Integer>();
		ArrayList<Double> blindList = new ArrayList<Double>();
		ArrayList<Double> DCList = new ArrayList<Double>();
		
		long t0=0,t1=0,t2=0;
		
		for(int n=MIN_N; n<MAX_N; n=increaseSize(n)){
			System.out.print("\nRunning for N = " + n);
			for(int i=1; i<REPEAT; i++){
				System.out.print(".");
				int[] a = generateRandomArray(n);
				
				t0 = System.currentTimeMillis();
				InversionCount.bruteForce(a);
				t1 = System.currentTimeMillis();
				InversionCount.sortAndCount(a);
				t2 = System.currentTimeMillis();
			}
			
			sizeList.add(n);
			blindList.add( (double)(t1-t0) / (double)(REPEAT) );
			DCList.add( (double)(t2-t1) / (double)(REPEAT) );
		}
		
//		System.out.println(sizeList);
//		System.out.println(blindList);
//		System.out.println(DCList);
		
		System.out.println("\n--------------------------------------------------------\n");
		
		PrintWriter writer;
		try {
			writer = new PrintWriter("plot_data.csv", "UTF-8");
		
			System.out.println("Size, Blind, D&C");
			writer.println("Size, Blind, D&C");
			
			
			for(int j=0; j<sizeList.size(); j++){
				System.out.println(sizeList.get(j) + ", " + blindList.get(j) + ", " + DCList.get(j));
				writer.println(sizeList.get(j) + ", " + blindList.get(j) + ", " + DCList.get(j));
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
	
	private static int[] generateRandomArray(int n) {
		int[] a = new int[n];
		for(int i=0; i<n; i++){
			a[i] = (int) Math.floor(MAX_N * Math.random());
		}
		
		
		return a;
	}

	private static int increaseSize(int n) {
		int output;
		
		if( n < 10)
			output = n + 1;
		else if( n < 100)
			output = n + 5;
		else
			output = 105 * n / 100;
		
		return output;
	}
}
