package com.example.matin_noohnezhad_assignment_2.activities;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.Spinner;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.Task;
import com.example.matin_noohnezhad_assignment_2.TaskType;

import java.util.Date;

public class NewTaskActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_new_task);
    }

    public void newTaskSave(View view) {

        EditText taskNameEt = findViewById(R.id.task_name_et);
        EditText dueDateEt = findViewById(R.id.due_date_et);
        EditText dueTimeHourEt = findViewById(R.id.hour_et);
        EditText dueTimeMinuteEt = findViewById(R.id.minute_et);
        Spinner typeSpinner = findViewById(R.id.type_spinner);
        EditText descriptionEt = findViewById(R.id.description_et);

        String type = typeSpinner.getSelectedItem().toString();
        TaskType taskType = TaskType.findTypeByString(type);
        String title = taskNameEt.getText().toString();
        String description = descriptionEt.getText().toString();
        String dueHour = dueTimeHourEt.getText().toString();
        String dueMinute = dueTimeMinuteEt.getText().toString();

        Task newTask = new Task(taskType, title, description, new Date(), dueHour, dueMinute);
        Task.addToDueTasks(newTask);
    }

}
