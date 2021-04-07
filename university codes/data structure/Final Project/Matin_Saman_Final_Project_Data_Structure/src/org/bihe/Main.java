package org.bihe;

public class Main {

    public static void main(String[] args) {
        MainAlgorithms.findBestPath(MainAlgorithms.floydWarshallAlgorithm(HelperMethods.stringParser(FileIO.readFile("arbi .txt"))));
    }


}
