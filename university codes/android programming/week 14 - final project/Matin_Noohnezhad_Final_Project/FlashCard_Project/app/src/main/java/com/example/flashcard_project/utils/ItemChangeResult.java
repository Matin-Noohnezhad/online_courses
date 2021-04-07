package com.example.flashcard_project.utils;

import java.util.List;


public class ItemChangeResult<T> {

    private List<T> items;
    private T item;

    private Action action;
    private Error error;


    public ItemChangeResult(List<T> items, T item, Action action, Error error) {
        this.items = items;
        this.item = item;
        this.action = action;
        this.error = error;
    }

    @Override
    public String toString() {
        String desc = "Action: " + action;
        if (error != null) {
            desc += "\nError: " + error.getMessage();
            return desc;
        }
        if (item != null) {
            desc += "\nItem: " + item.toString();
        }
        if (items != null) {
            desc += "\nItems: ";
            for (T i : items) {
                desc += "\n" + i.toString();
            }
        }
        return desc;
    }

    public List<T> getItems() {
        return items;
    }

    public void setItems(List<T> items) {
        this.items = items;
    }

    public T getItem() {
        return item;
    }

    public void setItem(T item) {
        this.item = item;
    }

    public Action getAction() {
        return action;
    }

    public void setAction(Action action) {
        this.action = action;
    }

    public Error getError() {
        return error;
    }

    public void setError(Error error) {
        this.error = error;
    }
}
