package com.example.flashcard_project.async_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.flashcard_project.async_tasks.inputs.CardAsyncInputs;
import com.example.flashcard_project.db.DatabaseHelper;
import com.example.flashcard_project.db.dao.FlashCardDao;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.network.NetworkHelper;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;

import java.util.List;

public class CardOpAsyncTask extends AsyncTask<CardAsyncInputs, Void, ItemChangeResult<FlashCard>> {

    private Context context;
    private Action action;
    private ItemChangeListener<FlashCard> changeListener;
    private NetworkHelper networkHelper;
    private DatabaseHelper databaseHelper;
    private FlashCardDao flashCardDao;

    public CardOpAsyncTask(Context context, Action action, ItemChangeListener<FlashCard> changeListener) {
        this.context = context;
        this.action = action;
        this.changeListener = changeListener;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        networkHelper = NetworkHelper.getInstance(context);
        databaseHelper = DatabaseHelper.getInstance(context);
        flashCardDao = databaseHelper.flashCardDao();
    }

    @Override
    protected ItemChangeResult<FlashCard> doInBackground(CardAsyncInputs... inputs) {
        if (inputs == null && inputs.length == 0) {
            return null;
        }
        CardAsyncInputs input = inputs[0];
        //
        ItemChangeResult<FlashCard> result = new ItemChangeResult<>(null, null, null, null);
        result.setAction(action);
        switch (action) {
            case INSERT_NET:
                networkHelper.addNewCard(input.getCard(), input.getUserToken(), changeListener);
                break;
            case GET_ALL_NET:
                networkHelper.getAllTask(input.getUserToken(),input.getOffset(), changeListener);
                break;
            case GET_ALL_DB:
                List<FlashCard> cards = flashCardDao.getAllByUID(input.getUserId());
                result.setItems(cards);
                break;
            case INSERT_ALL_DB:
                if (input.getCards() != null)
                    flashCardDao.insertAllFlashCard(input.getCards());
                result.setItems(input.getCards());
                break;
            case DELETE_ALL:
                flashCardDao.deleteAll();
                break;
            case INSERT_DB:
                flashCardDao.insert(input.getCard());
                result.setItem(input.getCard());
                break;
            case GET_CARDS_BY_UID_AND_LEVEL:
                List<FlashCard> cards2 = flashCardDao.getCardsByUidAndLevel(input.getUserId(), input.getLevel());
                result.setItems(cards2);
                break;
            case GET_ALL_BY_IDS:
                List<FlashCard> cards3 = flashCardDao.getAllByIds(input.getFcids());
                result.setItems(cards3);
                break;
            //TODO
        }
        return result;
    }

    @Override
    protected void onPostExecute(ItemChangeResult<FlashCard> flashCardItemChangeResult) {
        super.onPostExecute(flashCardItemChangeResult);
        //In network operation itemChanged() method called in NetworkHelper class.
        if (action == Action.INSERT_NET ||
                action == Action.GET_ALL_NET) {
            return;
        }
        changeListener.itemChanged(flashCardItemChangeResult);
    }
}
