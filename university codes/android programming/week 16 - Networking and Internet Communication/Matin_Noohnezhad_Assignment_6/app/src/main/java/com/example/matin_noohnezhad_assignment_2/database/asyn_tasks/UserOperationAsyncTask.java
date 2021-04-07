package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.matin_noohnezhad_assignment_2.database.AppDatabase;
import com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs.UserOperationInput;
import com.example.matin_noohnezhad_assignment_2.database.dao.UserDao;
import com.example.matin_noohnezhad_assignment_2.user.User;
import com.example.matin_noohnezhad_assignment_2.utils.Action;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeListener;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeResult;

import java.util.List;

public class UserOperationAsyncTask extends AsyncTask<UserOperationInput, Void, ItemChangeResult<User>> {

    private Context context;
    private Action action;
    private ItemChangeListener changeListener;
    private AppDatabase appDatabase;
    private UserDao userDao;

    public UserOperationAsyncTask(Context context, Action action, ItemChangeListener changeListener) {
        this.context = context;
        this.action = action;
        this.changeListener = changeListener;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        appDatabase = AppDatabase.getInstance(context);
        userDao = appDatabase.userDao();
    }

    @Override
    protected ItemChangeResult<User> doInBackground(UserOperationInput... userOperationInputs) {
        if (userOperationInputs.length == 0) {
            return null;
        }
        UserOperationInput input = userOperationInputs[0];
        ItemChangeResult<User> changeResult = null;
        switch (action) {
            case INSERT_DB:
                userDao.insert(input.getUser());
                changeResult = new ItemChangeResult<>(null, input.getUser(), Action.INSERT_DB, null);
                break;
            case CHECK_USERNAME_EXISTS_FOR_SIGN_IN_DB:
                /*input.getId() is the username!!!*/
                List<User> users = userDao.getUser(input.getId());
                changeResult = new ItemChangeResult<>(users, null, Action.CHECK_USERNAME_EXISTS_FOR_SIGN_IN_DB, null);
                break;
            case CHECK_USERNAME_EXISTS_FOR_SIGN_UP_DB:
                List<User> users2 = userDao.getUser(input.getId());
                changeResult = new ItemChangeResult<>(users2, null, Action.CHECK_USERNAME_EXISTS_FOR_SIGN_UP_DB, null);
                break;
            case UPDATE_DB:
                userDao.updateUserToken(input.getUser().getUid(), input.getUser().getUserToken());
                changeResult = new ItemChangeResult<>(null, input.getUser(), Action.UPDATE_DB, null);
                break;
        }
        return changeResult;
    }

    @Override
    protected void onPostExecute(ItemChangeResult<User> userItemChangeResult) {
        super.onPostExecute(userItemChangeResult);
        changeListener.itemChanged(userItemChangeResult);
    }
}
