package org.bihe;

public class MainAlgorithms {


    public static ConversionInfo[][] floydWarshallAlgorithm(ConversionInfo[][] conversionInfos) {

        ConversionInfo[][] finalResult = null;

        for (int k = 0; k < conversionInfos.length; k++) {
            for (int i = 0; i < conversionInfos.length; i++) {
                for (int j = 0; j < conversionInfos.length; j++) {
                    if (i != j && i != k && k != j) {
//                        System.out.println(k);
//                        System.out.println(i);
//                        System.out.println(j);
                        ConversionInfo a = conversionInfos[i][j];
                        ConversionInfo b = conversionInfos[i][k];
                        ConversionInfo c = conversionInfos[k][j];

                        if (a.getSumPathLog() < (b.getSumPathLog() + c.getSumPathLog())) {
                            //handle this problem that we dont have a cycle in our path
                            if (!(b.getShortestPath().contains("-->" + String.valueOf(i) + "-->") || b.getShortestPath().contains("-->" + String.valueOf(j) + "-->") || c.getShortestPath().contains("-->" + String.valueOf(i) + "-->") || c.getShortestPath().contains("-->" + String.valueOf(j) + "-->"))) {
                                //still we have to handle the problem of cycle in our path
                                String[] arr = b.getShortestPath().split("-->");
                                String[] arr2 = c.getShortestPath().split("-->");
                                boolean status = false;
                                for (int s = 0; s < arr.length; s++) {
                                    for (int l = 0; l < arr2.length; l++) {
                                        if (arr[s].equals(arr2[l])) {
                                            status = true;
                                            s = arr.length;
                                            l = arr2.length;
                                        }
                                    }
                                }
                                if (!status) {
                                    //test
//                                    if (i == 0 && j == 1) {
//                                        System.out.println("i = " + i);
//                                        System.out.println("j = " + j);
//                                        System.out.println("k = " + k);
//                                        System.out.println(a.getShortestPath());
//                                        System.out.println(b.getShortestPath());
//                                        System.out.println(c.getShortestPath());
//                                    }
                                    //
                                    a.setSumPathLog(b.getSumPathLog() + c.getSumPathLog());
                                    String firstPath = b.getShortestPath();
                                    String secondPath = c.getShortestPath();
                                    a.setShortestPath(firstPath + String.valueOf(k) + secondPath);
                                }
                            }
                        }
                    }
                }
            }
        }
//        for (int i = 0; i < 13; i++) {
//            System.out.println(String.valueOf(i) + conversionInfos[i][i + 1].getShortestPath() + String.valueOf(i + 1));
//        }
        finalResult = conversionInfos;
        return finalResult;
    }


    public static void findBestPath(ConversionInfo[][] conversionInfos) {

        double bestLog = 0;
        int a = 0;
        int b = 0;

        for (int i = 0; i < conversionInfos.length; i++) {
            for (int j = i; j < conversionInfos.length; j++) {
//                double first = conversionInfos[i][j].getSumPathLog();
//                double second = conversionInfos[j][i].getSumPathLog();
//                System.out.println(first+" "+second);
                double newLog = conversionInfos[i][j].getSumPathLog() + conversionInfos[j][i].getRateLog();
                if (newLog > bestLog) {
                    bestLog = newLog;
                    a = i;
                    b = j;
                }
            }
        }
        if (bestLog > 0) {
            System.out.println("Our profit is: " + Math.exp(bestLog));
            System.out.println(a);
            System.out.println(b);
            System.out.println("Our path is: " + String.valueOf(a) + conversionInfos[a][b].getShortestPath() + String.valueOf(b) + "-->" + String.valueOf(a));
        } else {
            System.out.println("we couldn't find any arbitrage in this sequence of conversion!!");
        }

    }

}
