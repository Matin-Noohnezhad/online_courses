package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

import com.example.matin_noohnezhad_assignment_2.database.AppDatabase;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs.TaskOperationInput;
import com.example.matin_noohnezhad_assignment_2.database.dao.TaskDao;
import com.example.matin_noohnezhad_assignment_2.task.Task;
import com.example.matin_noohnezhad_assignment_2.utils.Action;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeListener;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeResult;
import com.example.matin_noohnezhad_assignment_2.utils.TaskListHolder;

import java.util.ArrayList;
import java.util.List;

public class TaskOperationAsyncTask extends AsyncTask<TaskOperationInput, Void, ItemChangeResult<Task>> {

    private Context context;
    private Action action;
    private ItemChangeListener changeListener;
    private AppDatabase appDatabase;
    private TaskDao taskDao;

    public TaskOperationAsyncTask(Context context, Action action, ItemChangeListener changeListener) {
        this.context = context;
        this.action = action;
        this.changeListener = changeListener;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        appDatabase = AppDatabase.getInstance(context);
        taskDao = appDatabase.taskDao();
    }

    @Override
    protected ItemChangeResult<Task> doInBackground(TaskOperationInput... taskOperationInputs) {
        if (taskOperationInputs.length == 0) {
            return null;
        }
        TaskOperationInput input = taskOperationInputs[0];
        ItemChangeResult<Task> changeResult = null;
        switch (action) {
            case INSERT_DB:
                taskDao.insert(input.getTask());
                changeResult = new ItemChangeResult<>(null, input.getTask(), Action.INSERT_DB, null);
                break;
            case DELETE_DB:
                taskDao.delete(input.getId());
                changeResult = new ItemChangeResult<>(null, input.getTask(), Action.DELETE_DB, null);
                break;
            case UPDATE_DB:
                String tid = input.getTask().getTid();
                boolean isArchived = input.getTask().isArchived();
                taskDao.isArchiveUpdate(tid, isArchived);
                changeResult = new ItemChangeResult<>(null, input.getTask(), Action.UPDATE_DB, null);
                break;
            case SET_TASK:
                List<Task> tasks = taskDao.getAll(input.getId());
                fillTaskListHolder(tasks);
                break;
        }
        return changeResult;
    }

    @Override
    protected void onPostExecute(ItemChangeResult<Task> taskItemChangeResult) {
        super.onPostExecute(taskItemChangeResult);
        if (changeListener != null) {
            changeListener.itemChanged(taskItemChangeResult);
        }
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
        TaskListHolder.getInstance().setArchivedTaskList(archivedTasks);
        TaskListHolder.getInstance().setDueTaskList(dueTasks);
    }
}
