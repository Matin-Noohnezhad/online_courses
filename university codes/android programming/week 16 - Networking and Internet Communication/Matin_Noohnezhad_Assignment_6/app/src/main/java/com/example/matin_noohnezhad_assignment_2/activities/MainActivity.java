package com.example.matin_noohnezhad_assignment_2.activities;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.ValidationAnswer;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.TaskOperationAsyncTask;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.UserOperationAsyncTask;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs.TaskOperationInput;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs.UserOperationInput;
import com.example.matin_noohnezhad_assignment_2.network.NetworkHelper;
import com.example.matin_noohnezhad_assignment_2.task.Task;
import com.example.matin_noohnezhad_assignment_2.user.User;
import com.example.matin_noohnezhad_assignment_2.utils.Action;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeListener;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeResult;
import com.example.matin_noohnezhad_assignment_2.utils.TaskListHolder;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private User userTryToEnter;
    private UserChangeListener userChangeListener = new UserChangeListener();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void goToSignUpActivity(View view) {
//        Intent intent = new Intent(this, SignUpActivity.class);
        Intent intent = new Intent(this, SignUpActivity.class);
        startActivity(intent);
    }

    public void signInClicked(View view) {
        int duration = Toast.LENGTH_SHORT;
        ValidationAnswer va = validation();
        if (va.isValid()) {
            userTryToEnter = va.getUser();
            if (NetworkHelper.getInstance(this).isNetworkConnected()) {
                NetworkHelper.getInstance(this).userSignIn(userTryToEnter.getEmail(), userTryToEnter.getPassword(), userChangeListener);
            } else {
                UserOperationAsyncTask checkAsync = new UserOperationAsyncTask(this, Action.CHECK_USERNAME_EXISTS_FOR_SIGN_IN_DB, userChangeListener);
                UserOperationInput input = new UserOperationInput(null, userTryToEnter.getEmail());
                checkAsync.execute(input);
            }
        } else {
            Toast toast = Toast.makeText(this, va.getToastMessage(), duration);
            toast.show();
        }
    }

    public ValidationAnswer validation() {
        ValidationAnswer validationAnswer;
        User user = new User();
        String toastMessage = "";
        boolean isValid = true;
        //
        EditText usernameEditText = findViewById(R.id.main_username_et);
        EditText passwordEditText = findViewById(R.id.main_password_et);
        //
        String username = usernameEditText.getText().toString();
        if (!(username.length() > 0)) {
            isValid = false;
            toastMessage = "username field shouldn't be empty!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        user.setEmail(username);
        //
        String password = passwordEditText.getText().toString();
        if (!(password.length() > 0)) {
            isValid = false;
            toastMessage = "password field shouldn't be empty!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        //
        user.setPassword(password);
        validationAnswer = new ValidationAnswer(isValid, user, toastMessage);
        return validationAnswer;
    }

    private class UserChangeListener implements ItemChangeListener<User> {

        @Override
        public void itemChanged(ItemChangeResult<User> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  userChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in userChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(MainActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }
            Action action = result.getAction();
            switch (action) {
                case CHECK_USERNAME_EXISTS_FOR_SIGN_IN_DB:
                    List<User> users = result.getItems();
                    if (users == null || users.size() == 0) {
                        Toast toast = Toast.makeText(getApplicationContext(), "no such username exists!!!", Toast.LENGTH_SHORT);
                        toast.show();
                    } else {
                        if (users.get(0).getPassword().equals(userTryToEnter.getPassword())) {
                            PageActivity.setCurrentUser(users.get(0));
                            TaskOperationAsyncTask taskOperationAsync = new TaskOperationAsyncTask(getApplicationContext(), Action.SET_TASK, null);
                            TaskOperationInput input = new TaskOperationInput(null, users.get(0).getUid());
                            taskOperationAsync.execute(input);
                            Intent intent = new Intent(getApplicationContext(), PageActivity.class);
                            startActivity(intent);
                        } else {
                            Toast toast = Toast.makeText(getApplicationContext(), "password is wrong!!!", Toast.LENGTH_SHORT);
                            toast.show();
                        }
                    }
                    break;
                case SIGN_IN_NET:
                    User signedInUser = result.getItem();
                    UserOperationAsyncTask userOperationAsync = new UserOperationAsyncTask(MainActivity.this, Action.UPDATE_DB, this);
                    UserOperationInput input = new UserOperationInput(signedInUser, null);
                    userOperationAsync.execute(input);
                    break;
                case UPDATE_DB:
                    User updateUser = result.getItem();
                    PageActivity.setCurrentUser(updateUser);
                    NetworkHelper.getInstance(getApplicationContext()).getAllTask(updateUser.getUid(), updateUser.getUserToken(), new ItemChangeListener<Task>() {
                        @Override
                        public void itemChanged(ItemChangeResult<Task> result) {
                            if (result == null) {
                                Log.w(TAG, "Empty result received in taskChangeListener!");
                                return;
                            }
                            Log.d(TAG, "Result received in taskChangeListener: " + result.toString());
                            Error err = result.getError();
                            if (err != null) {
                                Toast.makeText(MainActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                                return;
                            }
                            Action action = result.getAction();
                            switch (action) {
                                case GET_ALL_NET:
                                    List<Task> tasks = result.getItems();
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
                                    break;
                            }
                        }
                    });
                    Intent intent = new Intent(getApplicationContext(), PageActivity.class);
                    startActivity(intent);
                    break;
            }
        }
    }
}
