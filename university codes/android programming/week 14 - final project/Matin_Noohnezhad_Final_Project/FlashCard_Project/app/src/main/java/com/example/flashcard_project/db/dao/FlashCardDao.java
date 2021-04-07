package com.example.flashcard_project.db.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import com.example.flashcard_project.beans.FlashCard;

import java.util.List;

@Dao
public interface FlashCardDao {

    @Insert
    long insert(FlashCard flashCard);

    @Insert
    void insertAllFlashCard(List<FlashCard> flashCards);

    @Query("select * from flashcard")
    List<FlashCard> getAll();

    @Query("select * from flashcard where fcId in " +
            "(select flash_card_id from usercardlevel where user_id =:uid)")
    List<FlashCard> getAllByUID(String uid);

    @Query("delete from flashcard")
    void deleteAll();

    @Query("select * from flashcard where fcId in (select flash_card_id from usercardlevel where user_id=:uid and level=:level)")
    List<FlashCard> getCardsByUidAndLevel(String uid, byte level);

    @Query("select * from flashcard where fcId in (:fcids)")
    List<FlashCard> getAllByIds(List<String> fcids);
}
