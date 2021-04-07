package org.bihe.probing;

public class LinearProbing implements Probing {

    private int hashTableSize;

    public LinearProbing(int hashTableSize) {
        this.hashTableSize = hashTableSize;
    }

    public int hashingFunction(int no, int probeNo) {
        return ((no % hashTableSize) + probeNo) % hashTableSize;
    }

}
