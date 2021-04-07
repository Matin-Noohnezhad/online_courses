package org.bihe;

public class HashTable {

    private int size;
    private Integer[] slot;

    public HashTable(int size) {
        this.size = size;
        slot = new Integer[this.size];
    }

    private boolean isFull(int index) {
        return (slot[index] != null);
    }

    public boolean insert(int no, int index) {
        if (isFull(index))
            return false;
        slot[index] = no;
        return true;
    }

    public boolean allSlotsAreFull() {
        boolean haveEmptySlot = false;
        for (int i = 0; i < size; i++) {
            if (slot[i] == null) haveEmptySlot = true;
            if (haveEmptySlot) break;
        }
        return !haveEmptySlot;
    }

    public void showTableContents() {
        System.out.println("__________________");
        System.out.println("| slots | inputs |");
        for (int i = 0; i < size; i++) {
            //------------------------------reshaping the table----------------------------
            int slotColumnLength = 5;
            int inputsColumnsLength = 6;
            String slotColumn = String.valueOf(i);
            boolean leftRightSpaceAdder = true;
            while (slotColumn.length() < slotColumnLength) {
                if (leftRightSpaceAdder) {
                    slotColumn = slotColumn + " ";
                    leftRightSpaceAdder = false;
                } else {
                    slotColumn = " " + slotColumn;
                    leftRightSpaceAdder = true;
                }
            }
            leftRightSpaceAdder = true;
            String inptusColumns = String.valueOf(slot[i]);
            while (inptusColumns.length() < inputsColumnsLength) {
                if (leftRightSpaceAdder) {
                    inptusColumns = inptusColumns + " ";
                    leftRightSpaceAdder = false;
                } else {
                    inptusColumns = " " + inptusColumns;
                    leftRightSpaceAdder = true;
                }
            }
            //------------------------------reshaping the table----------------------------
            System.out.println("__________________");
            System.out.println("| " + slotColumn + " | " + inptusColumns + " |");
        }
        System.out.println("__________________");
    }

    public int getSize() {
        return size;
    }
}
