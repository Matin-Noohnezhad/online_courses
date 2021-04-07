package com.example.flashcard_project.async_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.flashcard_project.async_tasks.inputs.UserAsyncInputs;
import com.example.flashcard_project.db.DatabaseHelper;
import com.example.flashcard_project.db.dao.UserDao;
import com.example.flashcard_project.beans.User;
import com.example.flashcard_project.network.NetworkHelper;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;

public class UserOpAsyncTask extends AsyncTask<UserAsyncInputs, Void, ItemChangeResult<User>> {

    private Context context;
    private Action action;
    private ItemChangeListener<User> changeListener;
    private NetworkHelper networkHelper;
    private DatabaseHelper databaseHelper;
    private UserDao userDao;

    public UserOpAsyncTask(Context context, Action action, ItemChangeListener changeListener) {
        this.context = context;
        this.action = action;
        this.changeListener = changeListener;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        networkHelper = NetworkHelper.getInstance(context);
        databaseHelper = DatabaseHelper.getInstance(context);
        userDao = databaseHelper.userDao();
    }

    @Override
    protected ItemChangeResult<User> doInBackground(UserAsyncInputs... inputs) {
        if (inputs == null || inputs.length == 0)
            return null;
        UserAsyncInputs input = inputs[0];
        //
        ItemChangeResult<User> result = new ItemChangeResult<>(null, null, null, null);
        result.setAction(action);
        switch (action) {
            case SIGN_IN_NET:
                networkHelper.userSignIn(input.getUser().getEmail(), input.getUser().getPassword(), changeListener);
                break;
            case SIGN_UP_NET:
                networkHelper.userSignUp(input.getUser().getEmail(), input.getUser().getPassword(), input.getUser().getName(), changeListener);
                break;
            case GET_USER_BY_ID:
                User user = userDao.getUserById(input.getUid());
                result.setItem(user);
                break;
            case UPDATE_DB:
                User u = input.getUser();
                userDao.updateUserToken(u.getUid(), u.getUserToken());
                result.setItem(u);
                break;
            case INSERT_DB:
                User u2 = input.getUser();
                userDao.insert(u2);
                result.setItem(u2);
                break;
            //TODO
        }
        return result;
    }

    @Override
    protected void onPostExecute(ItemChangeResult<User> userItemChangeResult) {
        super.onPostExecute(userItemChangeResult);
        //In network operation itemChanged() method called in NetworkHelper class.
        if (action == Action.SIGN_UP_NET ||
                action == Action.SIGN_IN_NET ||
                action == Action.CHECK_USER_TOKEN_VALIDITY_NET) {
            return;
        }
        changeListener.itemChanged(userItemChangeResult);
    }
}
