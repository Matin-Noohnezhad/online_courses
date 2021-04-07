package com.example.flashcard_project.activities;

import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
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

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class ReviewCardsActivity extends AppCompatActivity {

    private static final String TAG = "ReviewCardActivity";
    private TextToSpeech textToSpeech;
    private static final int SIZE_OF_SET = 10;
    private TextView wordTv;
    private TextView meaningTv;
    private CardChangeListener cardChangeListener = new CardChangeListener();
    private UserCardLevelChangeListener userCardLevelChangeListener = new UserCardLevelChangeListener();
    private List<FlashCard> cardSet;
    private List<UserCardLevel> userCardLevels;
    private int currentLevel = 1;
    private int numberOfNextCardInList = 0;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_review_cards);
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
        wordTv = findViewById(R.id.word_tv);
        meaningTv = findViewById(R.id.meaning_tv);
        //
        cardSet = new ArrayList<>();
        userCardLevels = new ArrayList<>();
        fillWordSet();
    }

    private void showNextCard() {
        if (numberOfNextCardInList + 1 > cardSet.size() || numberOfNextCardInList >= SIZE_OF_SET) {
            Intent intent = new Intent(this, AllCardsActivity.class);
            startActivity(intent);
            finish();
        }
        //
        numberOfNextCardInList++;
        showWord(cardSet.get(numberOfNextCardInList - 1));
        clearMeaning();
    }

    private void clearMeaning() {
        meaningTv.setText("");
    }

    private void showWord(FlashCard card) {
        wordTv.setText(card.getWord());
    }

    private void showMeaning(FlashCard card) {
        meaningTv.setText("Meaning: " + card.getDefinition() + "\n\nWord Usage: " + card.getWordUsage());
    }

    private void fillWordSet() {
        //get level 1  words
//        CardOpAsyncTask coat = new CardOpAsyncTask(this, Action.GET_CARDS_BY_UID_AND_LEVEL, cardChangeListener);
//        CardAsyncInputs input = new CardAsyncInputs(AllCardsActivity.getCurrentUser().getUid(), (byte) 1);
//        coat.execute(input);
        UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(this, Action.GET_USER_CARD_LEVELS_BY_LEVEL_AND_UID, userCardLevelChangeListener);
        UserCardLevelAsyncInputs input = new UserCardLevelAsyncInputs((byte) 1, AllCardsActivity.getCurrentUser().getUid());
        ucloat.execute(input);
    }

    private List<UserCardLevel> filterUserCardLevelsByLevel(List<UserCardLevel> list) {
        if (list == null || list.size() == 0) {
            return null;
        }
        if (currentLevel == 1) {
            return list;
        }
        if (currentLevel == 2) {
            List<UserCardLevel> filteredList = new ArrayList<>();
            //only the one which last reviewed is about more than 1 day ago
            for (UserCardLevel userCardLevel : list) {
                Date now = new Date();
                Date lastReveiwed = userCardLevel.getLastReveiwed();
                int a = now.getDate();
                int b = lastReveiwed.getDate();
//                LocalDateTime ldt = LocalDateTime.ofInstant(d.toInstant(), ZoneId.systemDefault());
//                LocalDateTime ldt2 = LocalDateTime.ofInstant(lastReveiwed.toInstant(), ZoneId.systemDefault());
//                Duration diff2 = Duration.between(ldt, ldt2);
                if (a != b)
                    filteredList.add(userCardLevel);
            }
            return filteredList;
        }
        if (currentLevel == 3) {
            List<UserCardLevel> filteredList = new ArrayList<>();
            //only the one which last reviewed is about more than 3 day ago
            for (UserCardLevel userCardLevel : list) {
                Date now = new Date();
                Date lastReveiwed = userCardLevel.getLastReveiwed();
                int a = (now.getDate() % 30);
                int b = (lastReveiwed.getDate() % 30);
                if (!(a != b && a != ((b + 1) % 30) && a != ((b + 2) % 30)))
                    filteredList.add(userCardLevel);
            }
            return filteredList;
        }
        return null;
    }

    public void pronounciationClicked(View view) {
        textToSpeech.speak(cardSet.get(numberOfNextCardInList - 1).getWord(), TextToSpeech.QUEUE_FLUSH, null);
    }

    public void showMeaningClicked(View view) {
        showMeaning(cardSet.get(numberOfNextCardInList - 1));
        //update review date of card
        UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(this, Action.UPDATE_LAST_REVIEWED, userCardLevelChangeListener);
        UserCardLevelAsyncInputs input = new UserCardLevelAsyncInputs(null, new UserCardLevel(AllCardsActivity.getCurrentUser().getUid(), cardSet.get(numberOfNextCardInList - 1).getFcId(), (byte) 0, new Date()));
        ucloat.execute(input);
    }

    public void iAnsweredRightClicked(View view) {
        //update level of card
        UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(this, Action.UPDATE_LEVEL_UP, userCardLevelChangeListener);
        UserCardLevelAsyncInputs input = new UserCardLevelAsyncInputs(null, new UserCardLevel(AllCardsActivity.getCurrentUser().getUid(), cardSet.get(numberOfNextCardInList - 1).getFcId(), (byte) 0, null));
        ucloat.execute(input);
        //update review date of card
        UserCardLevelOpAsyncTask ucloat2 = new UserCardLevelOpAsyncTask(this, Action.UPDATE_LAST_REVIEWED, userCardLevelChangeListener);
        UserCardLevelAsyncInputs input2 = new UserCardLevelAsyncInputs(null, new UserCardLevel(AllCardsActivity.getCurrentUser().getUid(), cardSet.get(numberOfNextCardInList - 1).getFcId(), (byte) 0, new Date()));
        ucloat2.execute(input2);
        //show next card in the list
        showNextCard();
    }

    public void iAnsweredWrongClicked(View view) {
        //update level of card
        UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(this, Action.UPDATE_LEVEL_DOWN, userCardLevelChangeListener);
        UserCardLevelAsyncInputs input = new UserCardLevelAsyncInputs(null, new UserCardLevel(AllCardsActivity.getCurrentUser().getUid(), cardSet.get(numberOfNextCardInList - 1).getFcId(), (byte) 0, null));
        ucloat.execute(input);
        //update review date of card
        UserCardLevelOpAsyncTask ucloat2 = new UserCardLevelOpAsyncTask(this, Action.UPDATE_LAST_REVIEWED, userCardLevelChangeListener);
        UserCardLevelAsyncInputs input2 = new UserCardLevelAsyncInputs(null, new UserCardLevel(AllCardsActivity.getCurrentUser().getUid(), cardSet.get(numberOfNextCardInList - 1).getFcId(), (byte) 0, new Date()));
        ucloat2.execute(input2);
        //show next card in the list
        showNextCard();
    }

    public void backToAllCardsClicked(View view) {
        Intent intent = new Intent(this, AllCardsActivity.class);
        startActivity(intent);
        finish();
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
                Toast.makeText(ReviewCardsActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }

            Action action = result.getAction();
            switch (action) {
                case GET_ALL_BY_IDS:
                    List<FlashCard> cards = result.getItems();
                    cardSet = cards;
                    //start showing cards
                    showNextCard();
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
                Toast.makeText(ReviewCardsActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }

            Action action = result.getAction();
            switch (action) {
                case GET_USER_CARD_LEVELS_BY_LEVEL_AND_UID:
                    List<UserCardLevel> ucls = result.getItems();
                    List<UserCardLevel> filteredList = filterUserCardLevelsByLevel(ucls);
                    if (filteredList != null)
                        userCardLevels.addAll(filteredList);
                    if (userCardLevels.size() < SIZE_OF_SET) {
                        //the size of set could be bigger so we search for next level words
                        currentLevel++;
                        if (currentLevel <= 3) {
                            UserCardLevelOpAsyncTask ucloat = new UserCardLevelOpAsyncTask(getApplicationContext(), Action.GET_USER_CARD_LEVELS_BY_LEVEL_AND_UID, this);
                            UserCardLevelAsyncInputs input = new UserCardLevelAsyncInputs((byte) currentLevel, AllCardsActivity.getCurrentUser().getUid());
                            ucloat.execute(input);
                        } else {
                            List<String> cardIds = UserCardLevel.getCardIdsFromList(userCardLevels);
                            CardOpAsyncTask coat = new CardOpAsyncTask(getApplicationContext(), Action.GET_ALL_BY_IDS, cardChangeListener);
                            CardAsyncInputs input = new CardAsyncInputs(cardIds);
                            coat.execute(input);
                        }
                    } else {
                        List<String> cardIds = UserCardLevel.getCardIdsFromList(userCardLevels);
                        CardOpAsyncTask coat = new CardOpAsyncTask(getApplicationContext(), Action.GET_ALL_BY_IDS, cardChangeListener);
                        CardAsyncInputs input = new CardAsyncInputs(cardIds);
                        coat.execute(input);
                    }
                    break;
                case UPDATE_LAST_REVIEWED:
                    //nothing to do
                    break;
                case UPDATE_LEVEL_UP:
                    //nothing to do
                    break;
                case UPDATE_LEVEL_DOWN:
                    //nothing to do
                    break;
            }
        }
    }
}
