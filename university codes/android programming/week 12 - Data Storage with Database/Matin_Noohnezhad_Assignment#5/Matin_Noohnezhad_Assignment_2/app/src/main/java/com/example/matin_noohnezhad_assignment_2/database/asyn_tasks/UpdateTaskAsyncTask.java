package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.matin_noohnezhad_assignment_2.activities.PageArchivedTasks;
import com.example.matin_noohnezhad_assignment_2.activities.PageDueTasks;
import com.example.matin_noohnezhad_assignment_2.database.AppDatabase;
import com.example.matin_noohnezhad_assignment_2.database.dao.TaskDao;
import com.example.matin_noohnezhad_assignment_2.task.Task;

public class UpdateTaskAsyncTask extends AsyncTask<Task, Void, Void> {

    private Context context;
    private AppDatabase appDatabase;
    private TaskDao taskDao;

    public UpdateTaskAsyncTask(Context context) {
        this.context = context;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        appDatabase = AppDatabase.getInstance(context);
        taskDao = appDatabase.taskDao();
    }

    @Override
    protected Void doInBackground(Task... tasks) {
        int tid = tasks[0].getTid();
        boolean isArchived = tasks[0].isArchived();
        taskDao.isArchiveUpdate(tid, !isArchived);
        return null;
    }

    @Override
    protected void onPostExecute(Void aVoid) {
        super.onPostExecute(aVoid);
        if (context instanceof PageArchivedTasks)
            ((PageArchivedTasks) context).afterTaskChangeStatus();
        else if (context instanceof PageDueTasks)
            ((PageDueTasks) context).afterTaskChangeStatus();
    }
}
