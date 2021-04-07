package com.example.matin_noohnezhad_assignment_2.database;

import android.content.Context;

import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.room.TypeConverters;

import com.example.matin_noohnezhad_assignment_2.database.dao.TaskDao;
import com.example.matin_noohnezhad_assignment_2.database.dao.UserDao;
import com.example.matin_noohnezhad_assignment_2.task.Task;
import com.example.matin_noohnezhad_assignment_2.user.User;

@Database(entities = {User.class, Task.class}, version = 1)
@TypeConverters({Converters.class})
public abstract class AppDatabase extends RoomDatabase {

    private static AppDatabase instance;

    public synchronized static AppDatabase getInstance(Context context) {
        if (instance == null)
            instance = Room.databaseBuilder(context, AppDatabase.class, "dbTest3").build();
        return instance;
    }

    public abstract UserDao userDao();

    public abstract TaskDao taskDao();

}
