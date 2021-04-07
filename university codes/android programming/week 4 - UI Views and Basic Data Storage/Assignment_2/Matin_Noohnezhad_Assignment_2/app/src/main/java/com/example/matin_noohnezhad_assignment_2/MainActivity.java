package com.example.matin_noohnezhad_assignment_2;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

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
        Context context = getApplicationContext();
//        CharSequence text = "Hello toast!";
        int duration = Toast.LENGTH_SHORT;
        ValidationAnswer va = validation();
        if (va.isValid()) {
            PreferenceManager preferenceManager = PreferenceManager.getInstance(context);
            User user = preferenceManager.getUser(va.getUser().getUsername(), "");
            if (user == null) {
                Toast toast = Toast.makeText(context, "no such username exists!!!", duration);
                toast.show();
            } else {
                Intent intent = new Intent(this, PageActivity.class);
                PageActivity.setCurrentUser(user);
                startActivity(intent);
            }
        } else {
            Toast toast = Toast.makeText(context, va.getToastMessage(), duration);
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

}
