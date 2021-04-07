package org.bihe.probing;

import org.bihe.probing.Probing;

public class QuadraticProbing implements Probing {

    private int hashTableSize;

    public QuadraticProbing(int hashTableSize) {
        this.hashTableSize = hashTableSize;
    }

    public int hashingFunction(int no, int probeNo) {
        return ((no % hashTableSize) + probeNo * probeNo) % hashTableSize;
    }

}
