package com.example.flashcard_project.db;

import android.content.Context;

import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.room.TypeConverters;

import com.example.flashcard_project.db.converters.Converters;
import com.example.flashcard_project.db.dao.FlashCardDao;
import com.example.flashcard_project.db.dao.UserCardLevelDao;
import com.example.flashcard_project.db.dao.UserDao;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.beans.User;
import com.example.flashcard_project.beans.UserCardLevel;

@Database(entities = {User.class, FlashCard.class, UserCardLevel.class}, version = 1)
@TypeConverters({Converters.class})
public abstract class DatabaseHelper extends RoomDatabase {

    private static DatabaseHelper instance;

    public synchronized static DatabaseHelper getInstance(Context context) {
        if (instance == null)
            instance = Room.databaseBuilder(context, DatabaseHelper.class, "flash_card_project").build();
        return instance;
    }

    public abstract UserDao userDao();

    public abstract FlashCardDao flashCardDao();

    public abstract UserCardLevelDao userCardLevelDao();

}
