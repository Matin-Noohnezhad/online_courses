package com.example.matin_noohnezhad_assignment_2.activities;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.ValidationAnswer;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.UserOperationAsyncTask;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs.UserOperationInput;
import com.example.matin_noohnezhad_assignment_2.network.NetworkHelper;
import com.example.matin_noohnezhad_assignment_2.user.Gender;
import com.example.matin_noohnezhad_assignment_2.user.User;
import com.example.matin_noohnezhad_assignment_2.utils.Action;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeListener;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeResult;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SignUpActivity extends AppCompatActivity {

    private static final String TAG = "SignUpActivity";
    private ValidationAnswer va;
    private UserChangeListener userChangeListener = new UserChangeListener();

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sign_up);
        userChangeListener.setContext(this);
    }

    public void confirmClicked(View view) {

        Context context = getApplicationContext();
        int duration = Toast.LENGTH_SHORT;
        va = validation();
        userChangeListener.setVa(va);
        if (va.isValid()) {
            NetworkHelper.getInstance(context).userSignUp(va.getUser(), userChangeListener);
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
        Gender g = Gender.toGender(gender);
        user.setGender(g);
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
        //username
        String username = usernameEditText.getText().toString();
        if (!(username.length() > 0)) {
            isValid = false;
            toastMessage = "username field shouldn't be empty!!!";
            validationAnswer = new ValidationAnswer(isValid, null, toastMessage);
            return validationAnswer;
        }
        user.setUsername(username);
        //
        validationAnswer = new ValidationAnswer(isValid, user, toastMessage);
        return validationAnswer;
    }

    boolean isEmailValid(CharSequence email) {
        return android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches();
    }

    private class UserChangeListener implements ItemChangeListener<User> {

        ValidationAnswer va;
        Context context;

        public void setContext(Context context) {
            this.context = context;
        }

        public void setVa(ValidationAnswer va) {
            this.va = va;
        }

        @Override
        public void itemChanged(ItemChangeResult<User> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  userChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in userChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(SignUpActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }
            Action action = result.getAction();
            switch (action) {
                case INSERT_DB:
//                    Intent intent = new Intent(this, MainActivity.class);
                    Intent intent = new Intent(context, MainActivity.class);
                    startActivity(intent);
                    break;
                case CHECK_USERNAME_EXISTS_FOR_SIGN_UP_DB:
//                    List<User> users = result.getItems();
//                    if (users == null || users.size() == 0) {
//                        UserOperationAsyncTask insertAsync = new UserOperationAsyncTask(context, Action.INSERT_DB, this);
//                        UserOperationInput input = new UserOperationInput(va.getUser(), null);
//                        insertAsync.execute(input);
//                        InsertUserAsyncTask iuat = new InsertUserAsyncTask(context);
//                        iuat.execute(va.getUser());
//                    } else {
//                        String toastMessage = "this username exists in database, choose another one!!!";
//                        Toast.makeText(context, toastMessage, Toast.LENGTH_SHORT).show();
//                    }
                    break;
                case SIGN_UP_NET:
                    User signedUpUser = result.getItem();
                    signedUpUser.setPassword(va.getUser().getPassword());
                    UserOperationAsyncTask insertAsync = new UserOperationAsyncTask(context, Action.INSERT_DB, this);
                    UserOperationInput input = new UserOperationInput(signedUpUser, null);
                    insertAsync.execute(input);
                    break;
            }

        }
    }

}
