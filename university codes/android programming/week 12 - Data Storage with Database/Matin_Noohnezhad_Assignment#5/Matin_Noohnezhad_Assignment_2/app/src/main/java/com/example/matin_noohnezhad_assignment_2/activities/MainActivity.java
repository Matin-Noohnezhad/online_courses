package com.example.matin_noohnezhad_assignment_2.activities;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.CheckUsernameExistsForSignInAsyncTask;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.SetTasksListAsyncTask;
import com.example.matin_noohnezhad_assignment_2.user.User;
import com.example.matin_noohnezhad_assignment_2.ValidationAnswer;

import java.util.List;

public class MainActivity extends AppCompatActivity {

    User userTryToEnter;

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
            CheckUsernameExistsForSignInAsyncTask cueat = new CheckUsernameExistsForSignInAsyncTask(this);
            cueat.execute(va.getUser().getUsername());
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
        user.setUsername(username);
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

    //This method calls from CheckUsrenameExistsAsyncTask
    public void afterGetUserFromDB(List<User> users) {
        if (users == null || users.size() == 0) {
            Toast toast = Toast.makeText(getApplicationContext(), "no such username exists!!!", Toast.LENGTH_SHORT);
            toast.show();
        } else {
            if (users.get(0).getPassword().equals(userTryToEnter.getPassword())) {
                Intent intent = new Intent(this, PageActivity.class);
                PageActivity.setCurrentUser(users.get(0));
                SetTasksListAsyncTask stlat = new SetTasksListAsyncTask(this);
                stlat.execute(users.get(0).getUid());
                startActivity(intent);
            } else {
                Toast toast = Toast.makeText(getApplicationContext(), "password is wrong!!!", Toast.LENGTH_SHORT);
                toast.show();
            }
        }
    }
}
