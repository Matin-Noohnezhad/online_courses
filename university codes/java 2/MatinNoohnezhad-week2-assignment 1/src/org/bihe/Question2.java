package org.bihe;

public class Question2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int[] arr = { 6, 3, 7, 9, 4, 1, 2 };
		// sort
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr.length - 1; j++) {
				if (arr[j] < arr[j + 1]) {
					int help = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = help;
				}
			}
		}
		if (arr.length > 2) {
			for (int i = 1; i < arr.length; i += 2) {
				int help = arr[i];
				arr[i] = arr[i + 1];
				arr[i + 1] = help;
			}
		}
		for (int i = 0; i < arr.length; i++) {
			System.out.print(arr[i]);
		}
	}

}
