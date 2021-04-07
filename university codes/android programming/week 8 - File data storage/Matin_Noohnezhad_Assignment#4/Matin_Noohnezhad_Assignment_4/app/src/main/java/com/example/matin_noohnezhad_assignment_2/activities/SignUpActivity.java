package com.example.matin_noohnezhad_assignment_2.activities;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.matin_noohnezhad_assignment_2.PreferenceManager;
import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.User;
import com.example.matin_noohnezhad_assignment_2.ValidationAnswer;
import com.example.matin_noohnezhad_assignment_2.activities.MainActivity;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SignUpActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sign_up);
    }

    public void confirmClicked(View view) {

        Context context = getApplicationContext();
//        CharSequence text = "Hello toast!";
        int duration = Toast.LENGTH_SHORT;
        ValidationAnswer va = validation();
        if (va.isValid()) {
            PreferenceManager preferenceManager = PreferenceManager.getInstance(context);
            User user = preferenceManager.getUser(va.getUser().getUsername(), "");
            if (user == null) {
                preferenceManager.putUser(va.getUser());
                Toast toast = Toast.makeText(context, "New user has been saved", duration);
                toast.show();
                Intent intent = new Intent(this, MainActivity.class);
                startActivity(intent);
            } else {
                Toast toast = Toast.makeText(context, "This username has exists!!!(choose another one)", duration);
                toast.show();
            }
        } else {
            Toast toast = Toast.makeText(context, va.getToastMessage(), duration);
            toast.show();
        }
    }

    private ValidationAnswer validation() {

        ValidationAnswer validationAnswer;
        User user = new User();
        String toastMessage = "";
        boolean isValid = true;
        //
        EditText usernameEditText = findViewById(R.id.username_et);
        EditText passwordEditText = findViewById(R.id.pass_et);
        EditText confirmPassEditText = findViewById(R.id.conf_pass_et);
        EditText emailEditText = findViewById(R.id.email_et);
        EditText mobileNoEditText = findViewById(R.id.mobile_no_et);
        Spinner languageSpinner = findViewById(R.id.lang_spinner);
        RadioGroup genderRadioGroup = findViewById(R.id.gender_rg);
        int selectedId = genderRadioGroup.getCheckedRadioButtonId();
        RadioButton genderRadioButton = findViewById(selectedId);
        CheckBox emailCheckBox = findViewById(R.id.email_cb);
        CheckBox smsCheckBox = findViewById(R.id.sms_cb);
        //username
        String username = usernameEditText.getText().toString();
        if (!(username.length() > 0)) {
            isValid = false;
            toastMessage = "username field shouldn't be empty!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        } else if (usernameExists(username)) {
            isValid = false;
            toastMessage = "this username exists choose another one!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        user.setUsername(username);
        //password
        String password = passwordEditText.getText().toString();
        if (!(password.length() > 0)) {
            isValid = false;
            toastMessage = "password field shouldn't be empty!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        user.setPassword(password);
        //confirm password
        String confirmPassword = confirmPassEditText.getText().toString();
        if (!(confirmPassword.length() > 0)) {
            isValid = false;
            toastMessage = "confirm password field shouldn't be empty!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        } else if (!(password.equals(confirmPassword))) {
            isValid = false;
            toastMessage = "confirm password and password should be the same!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        //email
        String email = emailEditText.getText().toString();
        if (!(email.length() > 0)) {
            isValid = false;
            toastMessage = "email field shouldn't be empty!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        } else if (!(isEmailValid(email))) {
            isValid = false;
            toastMessage = "email pattern is wrong!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        user.setEmail(email);
        //mobile number
        String mobileNo = mobileNoEditText.getText().toString();
        if (!(mobileNo.length() > 0)) {
            isValid = false;
            toastMessage = "mobile number field shouldn't be empty!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        String pattern = "^\\s*(?:\\+?(\\d{1,3}))?[-. (]*(\\d{3})[-. )]*(\\d{3})[-. ]*(\\d{4})(?: *x(\\d+))?\\s*$";
        Matcher m;
        Pattern r = Pattern.compile(pattern);
        m = r.matcher(mobileNo.trim());
        if (!m.find()) {
            isValid = false;
            toastMessage = "mobile number format is wrong!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        user.setMobileNo(mobileNo);
        //language spinner
        String language = languageSpinner.getSelectedItem().toString();
        user.setLanguage(language);
        //gender
        String gender = genderRadioButton.getText().toString();
        user.setGender(gender);
        //email send
        if (emailCheckBox.isChecked()) {
            user.setSendEmail(true);
        } else {
            user.setSendEmail(false);
        }
        //sms send
        if (smsCheckBox.isChecked()) {
            user.setSendSms(true);
        } else {
            user.setSendSms(false);
        }
        validationAnswer = new ValidationAnswer(isValid, user, toastMessage);
        return validationAnswer;
    }

    public boolean usernameExists(String username) {
        //complete it later
        return false;
    }

    boolean isEmailValid(CharSequence email) {
        return android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches();
    }

}
