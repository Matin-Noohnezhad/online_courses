package org.bihe;

import java.util.ArrayList;

public class HelperMethods {

//    public static void main(String[] args) {
//        stringParser(FileIO.readFile("arbi .txt"));
//
//    }


    public static ConversionInfo[][] stringParser(String input) {
        ConversionInfo[][] result = null;
        // 0 = 48 , 9 = 57 , . = 46
        ArrayList<String> numbers = new ArrayList<>();

        String number = "";
        for (int i = 0; i < input.length(); i++) {

            String s = input.substring(i, i + 1);

            if ((s.charAt(0) < 58 && s.charAt(0) > 47) || s.charAt(0) == 46) {
                number = number + s;
            } else {
                if (number != "" && number != null) {
                    numbers.add(number);
                    number = "";
                }
            }

        }
        if (number != "" && number != null) {
            numbers.add(number);
            number = "";
        }
        int size = (int) Math.sqrt(numbers.size());
        result = new ConversionInfo[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double d = Double.parseDouble(numbers.get((i * 14) + j));
                result[i][j] = new ConversionInfo(i, j, d);
            }
        }

        return result;
    }

}
