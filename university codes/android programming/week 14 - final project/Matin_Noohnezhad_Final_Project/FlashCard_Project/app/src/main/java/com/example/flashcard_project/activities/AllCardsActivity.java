package com.example.flashcard_project.activities;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.flashcard_project.R;
import com.example.flashcard_project.adapters.FlashCardAdapter;
import com.example.flashcard_project.async_tasks.CardOpAsyncTask;
import com.example.flashcard_project.async_tasks.inputs.CardAsyncInputs;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.beans.User;
import com.example.flashcard_project.shared_preferences.PreferenceManager;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.CardClickListener;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;

import java.util.ArrayList;
import java.util.List;

public class AllCardsActivity extends AppCompatActivity implements CardClickListener {

    private static final String TAG = "AllCardActivity";
    private static User currentUser;
    private RecyclerView recyclerView;
    private FlashCardAdapter cardAdapter;
    private CardChangeListener changeListener = new CardChangeListener();
    private static final int ADD_NEW_CARD_REQUEST = 1;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_all_cards);

        recyclerView = findViewById(R.id.recycler_all_cards);
        setUpRecyclerView();
    }

    private void setUpRecyclerView() {
        changeListener.setClickListener(this);
        CardOpAsyncTask cardOpAsyncTask = new CardOpAsyncTask(this, Action.GET_ALL_DB, changeListener);
        CardAsyncInputs input = new CardAsyncInputs(null, null, currentUser.getUid(), null);
        cardOpAsyncTask.execute(input);
    }

    public void reviewTodayCardsClicked(View view) {
        Intent intent = new Intent(this,ReviewCardsActivity.class);
        startActivity(intent);
        finish();
    }

    public void addNewCardClicked(View view) {
        Intent intent = new Intent(this, AddCardActivity.class);
        startActivityForResult(intent, ADD_NEW_CARD_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case ADD_NEW_CARD_REQUEST:
                if (resultCode == Activity.RESULT_OK) {
                    String id = data.getStringExtra("id");
                    String word = data.getStringExtra("word");
                    String definition = data.getStringExtra("definition");
                    String usage = data.getStringExtra("usage");
                    //
                    FlashCard card = new FlashCard(id, word, definition, usage);
                    cardAdapter.addItem(card);
                }
                break;
        }
    }

    public static User getCurrentUser() {
        return currentUser;
    }

    public static void setCurrentUser(User currentUser) {
        AllCardsActivity.currentUser = currentUser;
    }


    @Override
    public void onCardClicked(FlashCard card) {
        CardPageActivity.setPageCard(card);
        Intent intent = new Intent(this, CardPageActivity.class);
        startActivity(intent);
    }

    public void logoutClicked(View view) {
        CardOpAsyncTask coat = new CardOpAsyncTask(this, Action.DELETE_ALL, changeListener);
        CardAsyncInputs input = new CardAsyncInputs();
        coat.execute(input);
    }

    private class CardChangeListener implements ItemChangeListener<FlashCard> {

        private CardClickListener clickListener;

        public void setClickListener(CardClickListener clickListener) {
            this.clickListener = clickListener;
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
                Toast.makeText(AllCardsActivity.this, err.getMessage(), Toast.LENGTH_LONG).show();
                return;
            }

            Action action = result.getAction();

            switch (action) {
                case GET_ALL_DB:
                    List<FlashCard> cards = result.getItems();
                    if (cards == null)
                        cards = new ArrayList<>();
                    cardAdapter = new FlashCardAdapter(getApplicationContext(), cards, clickListener);
                    recyclerView.setLayoutManager(new LinearLayoutManager(getApplicationContext()));
                    recyclerView.setAdapter(cardAdapter);
                    break;
                case DELETE_ALL:
                    PreferenceManager.getInstance(getApplicationContext()).putCurrentUserId(null);
                    Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                    startActivity(intent);
                    finish();
                    break;
            }


        }
    }
}
