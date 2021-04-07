package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks;

import android.content.Context;
import android.os.AsyncTask;

import androidx.room.Room;

import com.example.matin_noohnezhad_assignment_2.activities.PageActivity;
import com.example.matin_noohnezhad_assignment_2.database.AppDatabase;
import com.example.matin_noohnezhad_assignment_2.database.dao.TaskDao;
import com.example.matin_noohnezhad_assignment_2.task.Task;
import com.example.matin_noohnezhad_assignment_2.utils.TaskListHolder;

import java.util.ArrayList;
import java.util.List;

public class SetTasksListAsyncTask extends AsyncTask<Integer, Void, Void> {

    private Context context;
    private AppDatabase appDatabase;
    private TaskDao taskDao;

    public SetTasksListAsyncTask(Context context) {
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
        List<Task> tasks = taskDao.getAll(integers[0]);
        fillTaskListHolder(tasks);
        return null;
    }

    private void fillTaskListHolder(List<Task> tasks) {
        List<Task> dueTasks = new ArrayList<>();
        List<Task> archivedTasks = new ArrayList<>();
        for (Task t : tasks) {
            if (t.isArchived())
                archivedTasks.add(t);
            else
                dueTasks.add(t);
        }
        TaskListHolder.setArchivedTaskList(archivedTasks);
        TaskListHolder.setDueTaskList(dueTasks);
    }
}
