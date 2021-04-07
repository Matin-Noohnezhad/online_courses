package com.example.flashcard_project.db.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import com.example.flashcard_project.beans.UserCardLevel;

import java.util.Date;
import java.util.List;

@Dao
public interface UserCardLevelDao {

    @Insert
    void insertAll(List<UserCardLevel> userCardLevelList);

    @Insert
    long insert(UserCardLevel userCardLevel);

    @Query("update usercardlevel set level=level+1 where user_id=:uid and flash_card_id=:fcId and level<3")
    void updateLevelUpById(String uid, String fcId);

    @Query("update usercardlevel set level=1 where user_id=:uid and flash_card_id=:fcId")
    void updateLevelDownById(String uid, String fcId);

    @Query("update usercardlevel set last_reviewed=:lastReviewed where user_id=:uid and flash_card_id=:fcid")
    void updateLastReviewed(String uid, String fcid, Date lastReviewed);

    @Query("select * from usercardlevel where user_id=:uid and level=:level")
    List<UserCardLevel> getUserCardLevelsByIdAndLevel(String uid, byte level);
}
