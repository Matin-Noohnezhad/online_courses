package com.example.matin_noohnezhad_assignment_2.activities;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.TaskOperationAsyncTask;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs.TaskOperationInput;
import com.example.matin_noohnezhad_assignment_2.network.NetworkHelper;
import com.example.matin_noohnezhad_assignment_2.task.Task;
import com.example.matin_noohnezhad_assignment_2.task.TaskAdapter;
import com.example.matin_noohnezhad_assignment_2.task.TaskClickListener;
import com.example.matin_noohnezhad_assignment_2.utils.Action;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeListener;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeResult;
import com.example.matin_noohnezhad_assignment_2.utils.TaskListHolder;

public class PageArchivedTasks extends AppCompatActivity implements TaskClickListener {

    private static final String TAG = "PageArchivedTask";
    private TaskAdapter taskAdapter;
    private RecyclerView recyclerView;
    private int deleteTaskPosition;
    private int changeStatusTaskPosition;
    private TaskChangeListener changeListener = new TaskChangeListener();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.archived_tasks_page);

        recyclerView = findViewById(R.id.recycler_view_2);
        setUpRecyclerView();
    }

    private void setUpRecyclerView() {
        taskAdapter = new TaskAdapter(this, TaskListHolder.getInstance().getArchivedTaskList(), this, false);
        changeListener.setTaskAdapter(taskAdapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(taskAdapter);
    }

    @Override
    public void onTaskDelete(int position) {
        deleteTaskPosition = position;
        changeListener.setDeleteTaskPosition(deleteTaskPosition);
        NetworkHelper.getInstance(this).removeTask(taskAdapter.getTaskByPosition(position), PageActivity.getCurrentUser().getUserToken(), changeListener);
    }

    @Override
    public void onTaskChangeStatus(int position) {
        changeStatusTaskPosition = position;
        changeListener.setChangeStatusTaskPosition(changeStatusTaskPosition);
        NetworkHelper.getInstance(this).updateIsArchivedTask(taskAdapter.getTaskByPosition(position), PageActivity.getCurrentUser().getUserToken(), changeListener);
    }

    private class TaskChangeListener implements ItemChangeListener<Task> {

        private int deleteTaskPosition;
        private int changeStatusTaskPosition;
        private TaskAdapter taskAdapter;

        public void setDeleteTaskPosition(int deleteTaskPosition) {
            this.deleteTaskPosition = deleteTaskPosition;
        }

        public void setChangeStatusTaskPosition(int changeStatusTaskPosition) {
            this.changeStatusTaskPosition = changeStatusTaskPosition;
        }

        public void setTaskAdapter(TaskAdapter taskAdapter) {
            this.taskAdapter = taskAdapter;
        }

        @Override
        public void itemChanged(ItemChangeResult<Task> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  taskChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in taskChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(PageArchivedTasks.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }
            Context context = getApplicationContext();
            Action action = result.getAction();
            switch (action) {
                case DELETE_DB:
                    taskAdapter.removeItem(deleteTaskPosition);
                    //
                    CharSequence text = "Task has been deleted!!!";
                    int duration = Toast.LENGTH_SHORT;
                    Toast toast = Toast.makeText(context, text, duration);
                    toast.show();
                    break;
                case UPDATE_DB:
                    Task t = taskAdapter.removeItem(changeStatusTaskPosition);
                    TaskListHolder.getInstance().addToDueTasks(result.getItem());
                    //
                    CharSequence text2 = "Task has been moved to due tasks!!!";
                    int duration2 = Toast.LENGTH_SHORT;
                    Toast toast2 = Toast.makeText(context, text2, duration2);
                    toast2.show();
                    break;
                case UPDATE_NET:
                    TaskOperationAsyncTask taskOperationAsync = new TaskOperationAsyncTask(context, Action.UPDATE_DB, this);
                    TaskOperationInput input = new TaskOperationInput(result.getItem(), null);
                    taskOperationAsync.execute(input);
                    break;
                case DELETE_NET:
                    TaskOperationAsyncTask taskOperationAsync2 = new TaskOperationAsyncTask(context, Action.DELETE_DB, changeListener);
                    TaskOperationInput input2 = new TaskOperationInput(null, result.getItem().getTid());
                    taskOperationAsync2.execute(input2);
                    break;
            }
        }
    }
}
