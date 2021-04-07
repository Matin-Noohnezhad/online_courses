package com.example.flashcard_project.activities;

import android.app.AlarmManager;
import android.app.PendingIntent;
import android.content.Intent;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NotificationCompat;

import com.example.flashcard_project.R;
import com.example.flashcard_project.async_tasks.CardOpAsyncTask;
import com.example.flashcard_project.async_tasks.UserOpAsyncTask;
import com.example.flashcard_project.async_tasks.inputs.CardAsyncInputs;
import com.example.flashcard_project.async_tasks.inputs.UserAsyncInputs;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.beans.User;
import com.example.flashcard_project.notifications.AlarmBroadcastReceiver;
import com.example.flashcard_project.notifications.NotificationHelper;
import com.example.flashcard_project.shared_preferences.PreferenceManager;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private int offsetForGettingListOFCards = 0;
    private List<FlashCard> listOfCardsRetrievedFromServer;
    private UserChangeListener userChangeListener = new UserChangeListener();
    private CardChangeListener cardChangeListener = new CardChangeListener();
    private EditText emailEt;
    private EditText passEt;
    private TextView errorTv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        //make notification every day with help of AlarmManger, BroadcastReceiver, Notification in android
        Intent intent = new Intent(this, AlarmBroadcastReceiver.class);
        PendingIntent pendingIntent = PendingIntent.getBroadcast(this.getApplicationContext(), 1234, intent, 0);
        AlarmManager alarmManager = (AlarmManager) getSystemService(ALARM_SERVICE);
        //cancel the last alarms (if we open the app before this)
        if (alarmManager != null) {
            alarmManager.cancel(pendingIntent);
        }
        //set alarm for 24 hours later and to repeat every 24 hours
        alarmManager.setInexactRepeating(AlarmManager.RTC_WAKEUP,
                (System.currentTimeMillis() + AlarmManager.INTERVAL_DAY),
                AlarmManager.INTERVAL_DAY, pendingIntent);
        //////////////////////////////////////////////////////////////////////////////////////////////////
        //check if the last user is still login or no.
        String uid = PreferenceManager.getInstance(this).getCurrentUserId();
        if (uid != null) {
            UserOpAsyncTask userOpAsyncTask = new UserOpAsyncTask(this, Action.GET_USER_BY_ID, userChangeListener);
            UserAsyncInputs input = new UserAsyncInputs(null, null, uid);
            userOpAsyncTask.execute(input);
        }
        //
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //
        emailEt = findViewById(R.id.email_et);
        passEt = findViewById(R.id.password_et);
        errorTv = findViewById(R.id.error_tv);
        //

    }

    private boolean isValid() {
        //
        if (emailEt.getText().toString() == null || emailEt.getText().toString().equals("")) {
            errorTv.setText("فیلد مربوط به ایمیل را پر کنید!!");
            return false;
        } else if (passEt.getText().toString() == null || passEt.getText().toString().equals("")) {
            errorTv.setText("فیلد مربوط به پسوورد را پر کنید!!");
            return false;
        } else if (!isEmailValid(emailEt.getText().toString())) {
            errorTv.setText("فرمت ایمیل وارد شده نامعتبر است!!");
            return false;
        }

        return true;
    }

    public void signInClicked(View view) {
        errorTv.setText("");
        if (isValid()) {
            String email = emailEt.getText().toString();
            String pass = passEt.getText().toString();
            userChangeListener.setPassword(pass);
            //sign in from server(because the last user-token in database may be expired)
            UserOpAsyncTask uoat = new UserOpAsyncTask(this, Action.SIGN_IN_NET, userChangeListener);
            User u = new User(null, null, pass, email, null);
            UserAsyncInputs input = new UserAsyncInputs(null, u, null);
            uoat.execute(input);
        }
    }

    public void goToSignUpClicked(View view) {
        Intent intent = new Intent(this, SignUpActivity.class);
        startActivity(intent);
        finish();
    }

    public static boolean isEmailValid(String email) {
        String expression = "^[\\w\\.-]+@([\\w\\-]+\\.)+[A-Z]{2,4}$";
        Pattern pattern = Pattern.compile(expression, Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(email);
        return matcher.matches();
    }

    private class UserChangeListener implements ItemChangeListener<User> {

        private String password;


        @Override
        public void itemChanged(ItemChangeResult<User> result) {

            if (result == null) {
                Log.w(TAG, "Empty result received in  UserChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in UserChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(MainActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }

            Action action = result.getAction();
            switch (action) {
                case GET_USER_BY_ID:
                    Log.d("getUserById", result.getItem().toString());
                    AllCardsActivity.setCurrentUser(result.getItem());
                    PreferenceManager.getInstance(getApplicationContext()).putCurrentUserId(result.getItem().getUid());
                    Intent intent = new Intent(getApplicationContext(), AllCardsActivity.class);
                    startActivity(intent);
                    finish();
                    break;
                case SIGN_IN_NET:
                    User signedInUser = result.getItem();
                    //
                    UserOpAsyncTask uoat = new UserOpAsyncTask(getApplicationContext(), Action.UPDATE_DB, this);
                    UserAsyncInputs input = new UserAsyncInputs(null, signedInUser, null);
                    uoat.execute(input);
                    break;
                case UPDATE_DB:
                    User signedInUserDB = result.getItem();
                    AllCardsActivity.setCurrentUser(signedInUserDB);
                    PreferenceManager.getInstance(getApplicationContext()).putCurrentUserId(signedInUserDB.getUid());
                    //
                    CardOpAsyncTask coat = new CardOpAsyncTask(getApplicationContext(), Action.GET_ALL_NET, cardChangeListener);
                    CardAsyncInputs input1 = new CardAsyncInputs(null, null, null, signedInUserDB.getUserToken());
                    cardChangeListener.setSignedInUserDB(signedInUserDB);
                    input1.setOffset(offsetForGettingListOFCards);
                    listOfCardsRetrievedFromServer = new ArrayList<>();
                    offsetForGettingListOFCards += 100;
                    coat.execute(input1);
                    break;
            }

        }

        public void setPassword(String password) {
            this.password = password;
        }
    }

    private class CardChangeListener implements ItemChangeListener<FlashCard> {

        private User signedInUserDB;

        @Override
        public void itemChanged(ItemChangeResult<FlashCard> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  CardChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in CardChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(MainActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }
            Action action = result.getAction();
            switch (action) {
                case GET_ALL_NET:
                    List<FlashCard> cards = result.getItems();
                    if (cards == null || cards.size() == 0) {
                        CardOpAsyncTask coat1 = new CardOpAsyncTask(getApplicationContext(), Action.INSERT_ALL_DB, this);
                        CardAsyncInputs input2 = new CardAsyncInputs(listOfCardsRetrievedFromServer, null, null, null);
                        coat1.execute(input2);
                        break;
                    }
                    listOfCardsRetrievedFromServer.addAll(cards);
                    if (cards.size() == 100) {
                        CardOpAsyncTask coat = new CardOpAsyncTask(getApplicationContext(), Action.GET_ALL_NET, this);
                        CardAsyncInputs input1 = new CardAsyncInputs(null, null, null, signedInUserDB.getUserToken());
                        input1.setOffset(offsetForGettingListOFCards);
                        offsetForGettingListOFCards += 100;
                        coat.execute(input1);
                    } else {
                        CardOpAsyncTask coat1 = new CardOpAsyncTask(getApplicationContext(), Action.INSERT_ALL_DB, this);
                        CardAsyncInputs input2 = new CardAsyncInputs(listOfCardsRetrievedFromServer, null, null, null);
                        coat1.execute(input2);
                    }
                    break;
                case INSERT_ALL_DB:
                    Intent intent2 = new Intent(getApplicationContext(), AllCardsActivity.class);
                    startActivity(intent2);
                    finish();
                    break;
            }
        }

        public void setSignedInUserDB(User signedInUserDB) {
            this.signedInUserDB = signedInUserDB;
        }
    }

}
