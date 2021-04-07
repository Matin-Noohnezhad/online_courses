package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.matin_noohnezhad_assignment_2.activities.MainActivity;
import com.example.matin_noohnezhad_assignment_2.activities.SignUpActivity;
import com.example.matin_noohnezhad_assignment_2.database.AppDatabase;
import com.example.matin_noohnezhad_assignment_2.database.dao.UserDao;
import com.example.matin_noohnezhad_assignment_2.user.User;

import java.util.ArrayList;
import java.util.List;

public class CheckUsernameExistsForSignUpAsyncTask extends AsyncTask<String, Void, List<User>> {

    private Context context;
    private AppDatabase appDatabase;
    private UserDao userDao;

    public CheckUsernameExistsForSignUpAsyncTask(Context context) {
        this.context = context;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        appDatabase = AppDatabase.getInstance(context);
        userDao = appDatabase.userDao();
    }

    @Override
    protected List<User> doInBackground(String... strings) {
        List<User> user = userDao.getUser(strings[0]);
        return user;
    }

    @Override
    protected void onPostExecute(List<User> users) {
        super.onPostExecute(users);
        //TODO
        ((SignUpActivity)context).afterCheckUsernameExistsInDB(users);
    }
}
