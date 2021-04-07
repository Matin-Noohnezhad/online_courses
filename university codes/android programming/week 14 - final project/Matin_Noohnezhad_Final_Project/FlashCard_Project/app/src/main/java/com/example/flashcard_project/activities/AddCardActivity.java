package com.example.flashcard_project.activities;

import android.app.Activity;
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
import com.example.flashcard_project.async_tasks.inputs.CardAsyncInputs;
import com.example.flashcard_project.async_tasks.inputs.UserCardLevelAsyncInputs;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.beans.UserCardLevel;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;

public class AddCardActivity extends AppCompatActivity {

    private static final String TAG = "AddCardActivity";
    private CardChangeListener cardChangeListener = new CardChangeListener();
    private UserCardLevelChangeListener userCardLevelChangeListener = new UserCardLevelChangeListener();
    private EditText wordEt;
    private EditText meaningEt;
    private EditText wordUsageEt;
    private TextView errorTv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_card);
        //
        wordEt = findViewById(R.id.word_et);
        meaningEt = findViewById(R.id.meaning_et);
        wordUsageEt = findViewById(R.id.word_usage_et);
        errorTv = findViewById(R.id.error_tv);
    }

    public void addFlashCardClicked(View view) {
        String word = wordEt.getText().toString();
        String meaning = meaningEt.getText().toString();
        String wordUsage = wordUsageEt.getText().toString();
        //
        if (word == null) {
            errorTv.setText("word field shouldn't be empty!!!");
            return;
        }
        if (meaning == null) {
            errorTv.setText("meaning field shouldn't be empty");
            return;
        }
        //
        FlashCard newCard = new FlashCard(word, meaning, wordUsage);
        CardOpAsyncTask coat = new CardOpAsyncTask(this, Action.INSERT_NET, cardChangeListener);
        CardAsyncInputs input = new CardAsyncInputs(null, newCard, null, AllCardsActivity.getCurrentUser().getUserToken());
        coat.execute(input);
        //
    }

    private class CardChangeListener implements ItemChangeListener<FlashCard> {

        @Override
        public void itemChanged(ItemChangeResult<FlashCard> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  CardChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in CardChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(AddCardActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }

            Action action = result.getAction();
            switch (action) {
                case INSERT_NET:
                    FlashCard insertedCard = result.getItem();
                    userCardLevelChangeListener.setInserted(insertedCard);
                    CardOpAsyncTask coat = new CardOpAsyncTask(getApplicationContext(), Action.INSERT_DB, this);
                    CardAsyncInputs input = new CardAsyncInputs(null, insertedCard, null, null);
                    coat.execute(input);
                    break;
                case INSERT_DB:
                    FlashCard insertedDBCard = result.getItem();
                    UserCardLevel ucl = UserCardLevel.createUserCardLevel(insertedDBCard.getFcId(), AllCardsActivity.getCurrentUser().getUid());
                    UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(getApplicationContext(), Action.INSERT_DB, userCardLevelChangeListener);
                    UserCardLevelAsyncInputs input2 = new UserCardLevelAsyncInputs(null, ucl);
                    ucloat.execute(input2);
                    break;
            }
        }
    }

    private class UserCardLevelChangeListener implements ItemChangeListener<UserCardLevel> {

        private FlashCard inserted;

        public void setInserted(FlashCard inserted) {
            this.inserted = inserted;
        }

        @Override
        public void itemChanged(ItemChangeResult<UserCardLevel> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  UserCardLevelChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in UserCardLevelChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(AddCardActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }

            Action action = result.getAction();
            switch (action) {
                case INSERT_DB:
                    Intent returnIntent = new Intent();
                    returnIntent.putExtra("id", inserted.getFcId());
                    returnIntent.putExtra("word", inserted.getWord());
                    returnIntent.putExtra("definition", inserted.getDefinition());
                    returnIntent.putExtra("usage", inserted.getWordUsage());
                    setResult(Activity.RESULT_OK, returnIntent);
                    finish();
                    break;
            }
        }
    }
}
