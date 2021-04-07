package com.example.flashcard_project.activities;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.flashcard_project.R;
import com.example.flashcard_project.async_tasks.CardOpAsyncTask;
import com.example.flashcard_project.async_tasks.UserCardLevelOpAsyncTask;
import com.example.flashcard_project.async_tasks.UserOpAsyncTask;
import com.example.flashcard_project.async_tasks.inputs.CardAsyncInputs;
import com.example.flashcard_project.async_tasks.inputs.UserAsyncInputs;
import com.example.flashcard_project.async_tasks.inputs.UserCardLevelAsyncInputs;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.beans.User;
import com.example.flashcard_project.beans.UserCardLevel;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SignUpActivity extends AppCompatActivity {

    private static final String TAG = "SignUpActivity";
    private UserChangeListener userChangeListener = new UserChangeListener();
    private CardChangeListener cardChangeListener = new CardChangeListener();
    private UserCardLevelChangeListener userCardLevelChangeListener = new UserCardLevelChangeListener();
    private int offsetForGettingListOFCards = 0;
    private List<FlashCard> listOfCardsRetrievedFromServer;
    private EditText nameEt;
    private EditText emailEt;
    private EditText passEt;
    private EditText confirmPassEt;
    private TextView errorTv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sign_up);
        //
        nameEt = findViewById(R.id.name_et);
        emailEt = findViewById(R.id.email_et);
        passEt = findViewById(R.id.pass_et);
        confirmPassEt = findViewById(R.id.confirm_pass_et);
        errorTv = findViewById(R.id.error_tv);
        //
    }

    private boolean isValid() {
        //
        String name = nameEt.getText().toString();
        String email = emailEt.getText().toString();
        String pass = passEt.getText().toString();
        String confirmPass = confirmPassEt.getText().toString();
        if (name == null || name.equals("")) {
            errorTv.setText("name field is empty!!!");
            return false;
        } else if (email == null || email.equals("")) {
            errorTv.setText("email field is empty!!!");
            return false;
        } else if (pass == null || pass.equals("")) {
            errorTv.setText("password field is empty!!!");
            return false;
        } else if (confirmPass == null || confirmPass.equals("")) {
            errorTv.setText("confirm password field is empty!!!");
            return false;
        } else if (!(pass.equals(confirmPass))) {
            errorTv.setText("password and confirm password is not equal!!!");
            return false;
        } else if (!isEmailValid(email)) {
            errorTv.setText("فرمت ایمیل وارد شده نامعتبر است!!");
            return false;
        }
        return true;
    }

    public void confirmClicked(View view) {
        errorTv.setText("");
        if (isValid()) {
            String name = nameEt.getText().toString();
            String email = emailEt.getText().toString();
            String pass = passEt.getText().toString();
            userChangeListener.setPassword(pass);
            //
            UserOpAsyncTask uoat = new UserOpAsyncTask(this, Action.SIGN_UP_NET, userChangeListener);
            User u = new User(null, name, pass, email, null);
            UserAsyncInputs input = new UserAsyncInputs(null, u, null);
            uoat.execute(input);
            //
        }
    }

    public static boolean isEmailValid(String email) {
        String expression = "^[\\w\\.-]+@([\\w\\-]+\\.)+[A-Z]{2,4}$";
        Pattern pattern = Pattern.compile(expression, Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(email);
        return matcher.matches();
    }

    private class UserChangeListener implements ItemChangeListener<User> {

        private String password;

        public void setPassword(String password) {
            this.password = password;
        }

        @Override
        public void itemChanged(ItemChangeResult<User> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  UserChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in UserChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(SignUpActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }
            Action action = result.getAction();
            switch (action) {
                case SIGN_UP_NET:
                    User signedUpUser = result.getItem();
                    signedUpUser.setPassword(password);
                    UserOpAsyncTask uoat = new UserOpAsyncTask(getApplicationContext(), Action.INSERT_DB, this);
                    UserAsyncInputs input = new UserAsyncInputs(null, signedUpUser, null);
                    uoat.execute(input);
                    break;
                case INSERT_DB:
                    User u = result.getItem();
                    cardChangeListener.setUid(u.getUid());
                    CardOpAsyncTask coat = new CardOpAsyncTask(getApplicationContext(), Action.GET_ALL_NET, cardChangeListener);
                    CardAsyncInputs input2 = new CardAsyncInputs(offsetForGettingListOFCards);
                    listOfCardsRetrievedFromServer = new ArrayList<>();
                    offsetForGettingListOFCards += 100;
                    coat.execute(input2);
                    break;
            }
        }

    }

    private class CardChangeListener implements ItemChangeListener<FlashCard> {

        private String uid;

        public void setUid(String uid) {
            this.uid = uid;
        }

        @Override
        public void itemChanged(ItemChangeResult<FlashCard> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  CardChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in CardChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(SignUpActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }
            Action action = result.getAction();
            switch (action) {
                case GET_ALL_NET:
                    List<FlashCard> cards = result.getItems();
                    if (cards == null || cards.size() == 0) {
                        List<UserCardLevel> list = UserCardLevel.convertFlashCardsToUserCardLevels(listOfCardsRetrievedFromServer, uid);
                        UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(getApplicationContext(), Action.INSERT_ALL_USER_CARD_LEVEL, userCardLevelChangeListener);
                        UserCardLevelAsyncInputs input = new UserCardLevelAsyncInputs(list, null);
                        ucloat.execute(input);
                        break;
                    }
                    listOfCardsRetrievedFromServer.addAll(cards);
                    int size = cards.size();
                    if (size == 100) {
                        CardOpAsyncTask coat = new CardOpAsyncTask(getApplicationContext(), Action.GET_ALL_NET, this);
                        CardAsyncInputs input2 = new CardAsyncInputs(offsetForGettingListOFCards);
                        offsetForGettingListOFCards += 100;
                        coat.execute(input2);
                    } else {
                        List<UserCardLevel> list = UserCardLevel.convertFlashCardsToUserCardLevels(listOfCardsRetrievedFromServer, uid);
                        UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(getApplicationContext(), Action.INSERT_ALL_USER_CARD_LEVEL, userCardLevelChangeListener);
                        UserCardLevelAsyncInputs input = new UserCardLevelAsyncInputs(list, null);
                        ucloat.execute(input);
                    }
                    break;
            }
        }
    }

    private class UserCardLevelChangeListener implements ItemChangeListener<UserCardLevel> {

        @Override
        public void itemChanged(ItemChangeResult<UserCardLevel> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  UserCardLevelChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in UserCardLevelChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(SignUpActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }
            Action action = result.getAction();
            switch (action) {
                case INSERT_ALL_USER_CARD_LEVEL:
                    Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                    startActivity(intent);
                    finish();
                    break;
            }
        }
    }
}
