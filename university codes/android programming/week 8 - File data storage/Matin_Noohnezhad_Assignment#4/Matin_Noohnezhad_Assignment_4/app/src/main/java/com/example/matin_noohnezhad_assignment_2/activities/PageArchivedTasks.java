package com.example.matin_noohnezhad_assignment_2.activities;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.Task;
import com.example.matin_noohnezhad_assignment_2.TaskAdapter;
import com.example.matin_noohnezhad_assignment_2.TaskClickListener;
import com.example.matin_noohnezhad_assignment_2.utils.TaskListHolder;

public class PageArchivedTasks extends AppCompatActivity implements TaskClickListener {

    private TaskAdapter taskAdapter;
    private RecyclerView recyclerView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.archived_tasks_page);

        recyclerView = findViewById(R.id.recycler_view_2);
        setUpRecyclerView();
    }

    private void setUpRecyclerView() {
        taskAdapter = new TaskAdapter(this, TaskListHolder.getArchivedTaskList(), this, false);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(taskAdapter);
    }

    @Override
    public void onTaskDelete(int position) {
        taskAdapter.removeItem(position);
        //
        Context context = getApplicationContext();
        CharSequence text = "Task has been deleted!!!";
        int duration = Toast.LENGTH_SHORT;
        Toast toast = Toast.makeText(context, text, duration);
        toast.show();
    }

    @Override
    public void onTaskChangeStatus(int position) {
        Task t = taskAdapter.removeItem(position);
        TaskListHolder.addToDueTasks(t);
        //
        Context context = getApplicationContext();
        CharSequence text = "Task has been moved to due tasks!!!";
        int duration = Toast.LENGTH_SHORT;
        Toast toast = Toast.makeText(context, text, duration);
        toast.show();
    }
}
