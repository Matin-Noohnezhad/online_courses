package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks;

import android.content.Context;
import android.os.AsyncTask;

import com.example.matin_noohnezhad_assignment_2.activities.SignUpActivity;
import com.example.matin_noohnezhad_assignment_2.database.AppDatabase;
import com.example.matin_noohnezhad_assignment_2.database.dao.UserDao;
import com.example.matin_noohnezhad_assignment_2.user.User;

public class InsertUserAsyncTask extends AsyncTask<User, Void, Boolean> {

    private Context context;
    private AppDatabase appDatabase;
    private UserDao userDao;

    public InsertUserAsyncTask(Context context) {
        this.context = context;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
        appDatabase = AppDatabase.getInstance(context);
        userDao = appDatabase.userDao();
    }

    @Override
    protected Boolean doInBackground(User... users) {
        long id = userDao.insert(users[0]);
        if (id < 0)
            return false;
        return true;
    }

    @Override
    protected void onPostExecute(Boolean aBoolean) {
        super.onPostExecute(aBoolean);
        ((SignUpActivity) context).afterInsert(aBoolean);
    }
}
