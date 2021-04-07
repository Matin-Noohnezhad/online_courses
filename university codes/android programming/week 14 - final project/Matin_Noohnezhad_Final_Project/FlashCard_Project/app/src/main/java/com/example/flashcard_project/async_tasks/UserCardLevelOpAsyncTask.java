package com.example.flashcard_project.async_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.flashcard_project.async_tasks.inputs.UserCardLevelAsyncInputs;
import com.example.flashcard_project.db.DatabaseHelper;
import com.example.flashcard_project.db.dao.UserCardLevelDao;
import com.example.flashcard_project.beans.UserCardLevel;
import com.example.flashcard_project.network.NetworkHelper;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;

import java.util.List;

public class UserCardLevelOpAsyncTask extends AsyncTask<UserCardLevelAsyncInputs, Void, ItemChangeResult<UserCardLevel>> {

    private Context context;
    private Action action;
    private ItemChangeListener<UserCardLevel> changeListener;
    private NetworkHelper networkHelper;
    private DatabaseHelper databaseHelper;
    private UserCardLevelDao userCardLevelDao;

    public UserCardLevelOpAsyncTask(Context context, Action action, ItemChangeListener changeListener) {
        this.context = context;
        this.action = action;
        this.changeListener = changeListener;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        networkHelper = NetworkHelper.getInstance(context);
        databaseHelper = DatabaseHelper.getInstance(context);
        userCardLevelDao = databaseHelper.userCardLevelDao();
    }

    @Override
    protected ItemChangeResult<UserCardLevel> doInBackground(UserCardLevelAsyncInputs... inputs) {
        if (inputs == null || inputs.length == 0)
            return null;
        UserCardLevelAsyncInputs input = inputs[0];
        //
        ItemChangeResult<UserCardLevel> result = new ItemChangeResult<>(null, null, null, null);
        result.setAction(action);
        switch (action) {
            case INSERT_ALL_USER_CARD_LEVEL:
                userCardLevelDao.insertAll(input.getItems());
                break;
            case INSERT_DB:
                userCardLevelDao.insert(input.getItem());
                break;
            case UPDATE_DB:
                userCardLevelDao.updateLastReviewed(input.getItem().getUId(), input.getItem().getFcId(), input.getItem().getLastReveiwed());
                break;
            case UPDATE_LAST_REVIEWED:
                userCardLevelDao.updateLastReviewed(input.getItem().getUId(), input.getItem().getFcId(), input.getItem().getLastReveiwed());
                break;
            case UPDATE_LEVEL_UP:
                userCardLevelDao.updateLevelUpById(input.getItem().getUId(), input.getItem().getFcId());
                break;
            case UPDATE_LEVEL_DOWN:
                userCardLevelDao.updateLevelDownById(input.getItem().getUId(), input.getItem().getFcId());
                break;
            case GET_USER_CARD_LEVELS_BY_LEVEL_AND_UID:
                List<UserCardLevel> list = userCardLevelDao.getUserCardLevelsByIdAndLevel(input.getUid(),input.getLevel());
                result.setItems(list);
            //TODO
        }
        return result;
    }

    @Override
    protected void onPostExecute(ItemChangeResult<UserCardLevel> result) {
        super.onPostExecute(result);
        changeListener.itemChanged(result);
    }
}
