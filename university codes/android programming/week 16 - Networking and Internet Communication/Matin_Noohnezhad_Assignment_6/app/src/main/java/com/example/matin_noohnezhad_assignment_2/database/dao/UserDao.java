package com.example.matin_noohnezhad_assignment_2.database.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Update;

import com.example.matin_noohnezhad_assignment_2.user.User;

import java.util.List;

@Dao
public interface UserDao {

    @Insert
    long insert(User user);

    @Query("select * from user where email=:email")
    List<User> getUser(String email);

    @Query("update user set user_token=:userToken where uid=:uid")
    void updateUserToken(String uid, String userToken);

}
