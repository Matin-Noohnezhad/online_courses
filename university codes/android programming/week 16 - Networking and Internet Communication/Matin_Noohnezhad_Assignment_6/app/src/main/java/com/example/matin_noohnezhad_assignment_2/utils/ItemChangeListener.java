package com.example.matin_noohnezhad_assignment_2.utils;

/* Can be used as callback to notify any listener interested in item change results;
   e.g. Activity getting db operation result from AsyncTask, or getting list selection from list adapter */
public interface ItemChangeListener<T> {

    public void itemChanged(ItemChangeResult<T> result);

}
