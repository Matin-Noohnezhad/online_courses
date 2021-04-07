package com.example.matin_noohnezhad_assignment_2.database.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import com.example.matin_noohnezhad_assignment_2.user.User;

import java.util.List;

@Dao
public interface UserDao {

    @Insert
    long insert(User user);

    @Query("select * from user where username=:username")
    List<User> getUser(String username);

}
