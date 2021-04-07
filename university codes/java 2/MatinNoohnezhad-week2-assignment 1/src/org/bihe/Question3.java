package org.bihe;

import java.util.Scanner;

public class Question3 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc = new Scanner(System.in);
		String numbers = sc.next();
		String[] arr = numbers.split(",");
		int[] arr2 = new int[arr.length];
		for (int i = 0; i < arr.length; i++) {
			arr2[i] = Integer.parseInt(arr[i]);
		}
		int index = 0;
		int answer = minimum(arr2, index);
		System.out.println(answer);
		sc.close();
	}

	public static int minimum(int[] arr, int index) {
		if (arr[index] < arr[index + 1]) {
			return index;
		}
		for (int i = index + 1; i < arr.length - 1; i++) {
			if (arr[i] < arr[i + 1]) {
				return minimum(arr, i);
			}
		}
		return arr.length - 1;

	}

}
