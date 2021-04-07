package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.matin_noohnezhad_assignment_2.activities.PageArchivedTasks;
import com.example.matin_noohnezhad_assignment_2.activities.PageDueTasks;
import com.example.matin_noohnezhad_assignment_2.database.AppDatabase;
import com.example.matin_noohnezhad_assignment_2.database.dao.TaskDao;

public class DeleteTaskAsyncTask extends AsyncTask<Integer, Void, Void> {

    private Context context;
    private AppDatabase appDatabase;
    private TaskDao taskDao;

    public DeleteTaskAsyncTask(Context context) {
        this.context = context;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        appDatabase = AppDatabase.getInstance(context);
        taskDao = appDatabase.taskDao();
    }

    @Override
    protected Void doInBackground(Integer... integers) {
        taskDao.delete(integers[0]);
        return null;
    }

    @Override
    protected void onPostExecute(Void aVoid) {
        super.onPostExecute(aVoid);
        if (context instanceof PageArchivedTasks)
            ((PageArchivedTasks) context).afterTaskDeleteFromDB();
        else if (context instanceof PageDueTasks)
            ((PageDueTasks) context).afterTaskDeleteFromDB();
    }
}
