package com.example.flashcard_project.db.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Update;

import com.example.flashcard_project.beans.User;

@Dao
public interface UserDao {

    @Insert
    long insert(User user);

    @Update
    void update(User user);

    @Query("select * from user where uid=:uid")
    User getUserById(String uid);

    @Query("select * from user where email=:email")
    User getUserByEmail(String email);

    @Query("update user set user_token=:userToken where uid=:uid")
    void updateUserToken(String uid, String userToken);
}
