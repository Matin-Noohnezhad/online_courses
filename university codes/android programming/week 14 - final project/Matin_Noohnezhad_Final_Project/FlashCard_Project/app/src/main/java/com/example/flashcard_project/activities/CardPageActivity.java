package com.example.flashcard_project.activities;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.flashcard_project.R;
import com.example.flashcard_project.async_tasks.UserCardLevelOpAsyncTask;
import com.example.flashcard_project.async_tasks.inputs.UserCardLevelAsyncInputs;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.beans.UserCardLevel;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;

import java.util.Date;
import java.util.Locale;

public class CardPageActivity extends AppCompatActivity {

    private static final String TAG = "CardPageActivity";
    private TextToSpeech textToSpeech;
    private userCardLevelChangeListener changeListener = new userCardLevelChangeListener();
    private static FlashCard pageCard;
    private TextView wordTv;
    private TextView meaningTv;
    private TextView wordUsageTv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_card_page);
        //
        textToSpeech = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != TextToSpeech.ERROR) {
                    textToSpeech.setLanguage(Locale.UK);
                }
            }
        });
        //
        setUpTextViews();
    }

    public void setUpTextViews() {
        wordTv = findViewById(R.id.word_tv);
        meaningTv = findViewById(R.id.meaning_tv);
        wordUsageTv = findViewById(R.id.word_usage_tv);
        //
        wordTv.setText(pageCard.getWord());
        meaningTv.setText("Meaning:\n\t\t\t" + pageCard.getDefinition());
        wordUsageTv.setText("Word Usage:\n\t\t\t" + pageCard.getWordUsage());
        //
        UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(this, Action.UPDATE_DB, changeListener);
        UserCardLevelAsyncInputs input = new UserCardLevelAsyncInputs(null, new UserCardLevel(AllCardsActivity.getCurrentUser().getUid(), pageCard.getFcId(), (byte) 0, new Date()));
        ucloat.execute(input);
    }

    public TextView getWordTv() {
        return wordTv;
    }

    public void setWordTv(TextView wordTv) {
        this.wordTv = wordTv;
    }

    public TextView getMeaningTv() {
        return meaningTv;
    }

    public void setMeaningTv(TextView meaningTv) {
        this.meaningTv = meaningTv;
    }

    public TextView getWordUsageTv() {
        return wordUsageTv;
    }

    public void setWordUsageTv(TextView wordUsageTv) {
        this.wordUsageTv = wordUsageTv;
    }

    public void pronounciationClicked(View view) {
        textToSpeech.speak(pageCard.getWord(), TextToSpeech.QUEUE_FLUSH, null);

    }

    public static FlashCard getPageCard() {
        return pageCard;
    }

    public static void setPageCard(FlashCard pageCard) {
        CardPageActivity.pageCard = pageCard;
    }

    private class userCardLevelChangeListener implements ItemChangeListener<UserCardLevel> {

        @Override
        public void itemChanged(ItemChangeResult<UserCardLevel> result) {
            if (result == null) {
                Log.w(TAG, "Empty result received in  userCardLevelChangeListener!");
                return;
            }
            Log.d(TAG, "Result received in userCardLevelChangeListener: " + result.toString());
            Error err = result.getError();
            if (err != null) {
                Toast.makeText(CardPageActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }

            Action action = result.getAction();
            switch (action) {
                case UPDATE_DB:
                    //nothing
                    break;
            }
        }
    }
}
