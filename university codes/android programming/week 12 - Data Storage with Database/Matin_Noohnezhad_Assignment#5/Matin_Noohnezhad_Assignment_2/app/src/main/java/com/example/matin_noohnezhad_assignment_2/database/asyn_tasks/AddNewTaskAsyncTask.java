package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.matin_noohnezhad_assignment_2.activities.NewTaskActivity;
import com.example.matin_noohnezhad_assignment_2.activities.PageActivity;
import com.example.matin_noohnezhad_assignment_2.database.AppDatabase;
import com.example.matin_noohnezhad_assignment_2.database.dao.TaskDao;
import com.example.matin_noohnezhad_assignment_2.task.Task;

public class AddNewTaskAsyncTask extends AsyncTask<Task, Void, Boolean> {

    private Context context;
    private AppDatabase appDatabase;
    private TaskDao taskDao;

    public AddNewTaskAsyncTask(Context context) {
        this.context = context;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        appDatabase = AppDatabase.getInstance(context);
        taskDao = appDatabase.taskDao();
    }

    @Override
    protected Boolean doInBackground(Task... tasks) {
        long id = taskDao.insert(tasks[0]);
        if (id < 0)
            return false;
        return true;
    }

    @Override
    protected void onPostExecute(Boolean aBoolean) {
        super.onPostExecute(aBoolean);
        ((NewTaskActivity) context).afterNewTaskAdded(aBoolean);
    }
}
