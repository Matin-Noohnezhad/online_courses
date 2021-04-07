package com.example.matin_noohnezhad_assignment_2.activities;

import android.app.DatePickerDialog;
import android.app.TimePickerDialog;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.TimePicker;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.TaskOperationAsyncTask;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs.TaskOperationInput;
import com.example.matin_noohnezhad_assignment_2.network.NetworkHelper;
import com.example.matin_noohnezhad_assignment_2.task.Task;
import com.example.matin_noohnezhad_assignment_2.task.TaskType;
import com.example.matin_noohnezhad_assignment_2.utils.Action;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeListener;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeResult;
import com.example.matin_noohnezhad_assignment_2.utils.TaskListHolder;

import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;

public class NewTaskActivity extends AppCompatActivity {

    private static final String TAG = "NewTaskActivity";
    private EditText et;
    private EditText hour_et;
    private EditText minute_et;
    private Task newTask;
    private TaskChangeListener changeListener = new TaskChangeListener();

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
        String date = dueDateEt.getText().toString();
        String[] arrDate = date.split("/");
        if (type.equals("") || type == null || title.equals("") || title == null || dueHour.equals("") || dueHour == null || dueMinute.equals("") || dueMinute == null || date.equals("") || date == null) {
            showError("you must fill all the fields");
        } else {
            int day = Integer.parseInt(arrDate[0]);
            int month = Integer.parseInt(arrDate[1]);
            int year = Integer.parseInt(arrDate[2]);
            String uid = PageActivity.getCurrentUser().getUid();
            newTask = new Task(taskType, title, description, new GregorianCalendar(year, month - 1, day).getTime(), dueHour, dueMinute, uid);
            //
            NetworkHelper.getInstance(this).addNewTask(newTask, PageActivity.getCurrentUser().getUserToken(), changeListener);
        }
    }
/*
    public void afterNewTaskAdded(boolean b) {
        if (!b) {
            showError("Task doesn't add to database successfully");
        } else {
            TaskListHolder.addToDueTasks(newTask);
            Intent intent = new Intent(getApplicationContext(), PageDueTasks.class);
            startActivity(intent);
        }
    }
*/

    public void datePicking(View view) {
        et = (EditText) findViewById(R.id.due_date_et);
        Date d = new Date();
        int year = d.getYear() + 1900;
        int month = d.getMonth();
        int dayOfMonth = d.getDate();

        DatePickerDialog dpd = new DatePickerDialog(NewTaskActivity.this, new DatePickerDialog.OnDateSetListener() {
            @Override
            public void onDateSet(DatePicker view, int mYear, int mMonth, int mDayOfMonth) {
                et.setText(mDayOfMonth + "/" + (mMonth + 1) + "/" + mYear);
            }
        }, year, month, dayOfMonth);
        dpd.show();
    }

    public void timePicking(View view) {
        hour_et = (EditText) findViewById(R.id.hour_et);
        minute_et = (EditText) findViewById(R.id.minute_et);
        Calendar c = GregorianCalendar.getInstance();
        int hourOfDay = c.get(GregorianCalendar.HOUR_OF_DAY);
        int minute = c.get(GregorianCalendar.MINUTE);

        TimePickerDialog tpd = new TimePickerDialog(NewTaskActivity.this, new TimePickerDialog.OnTimeSetListener() {
            @Override
            public void onTimeSet(TimePicker view, int mHourOfDay, int mMinute) {
                hour_et.setText(String.valueOf(mHourOfDay));
                minute_et.setText(String.valueOf(mMinute));
            }

        }, hourOfDay, minute, true);
        tpd.show();
    }

    public void showError(String message) {
        TextView error_et = findViewById(R.id.error_tv);
        error_et.setText(message);
    }

    private class TaskChangeListener implements ItemChangeListener<Task> {

        @Override
        public void itemChanged(ItemChangeResult<Task> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  taskChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in taskChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(NewTaskActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }
            Action action = result.getAction();
            switch (action) {
                case INSERT_DB:
                    TaskListHolder.getInstance().addToDueTasks(result.getItem());
                    Intent intent = new Intent(getApplicationContext(), PageDueTasks.class);
                    startActivity(intent);
                    break;
                case INSERT_NET:
                    Task insertedTask = result.getItem();
                    TaskOperationAsyncTask taskOperationAsync = new TaskOperationAsyncTask(getApplicationContext(), Action.INSERT_DB, this);
                    TaskOperationInput input = new TaskOperationInput(insertedTask, null);
                    taskOperationAsync.execute(input);
                    break;
            }
        }
    }

}
