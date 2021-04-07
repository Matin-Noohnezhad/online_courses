package com.example.flashcard_project.beans;

import androidx.annotation.NonNull;
import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

@Entity
public class UserCardLevel {

    @PrimaryKey(autoGenerate = true)
    @NonNull
    private int id;
    @ColumnInfo(name = "user_id")
    private String uId;
    @ColumnInfo(name = "flash_card_id")
    private String fcId;
    @ColumnInfo
    private byte level;
    @ColumnInfo(name = "last_reviewed")
    private Date lastReveiwed;

    public UserCardLevel(int id, String uId, String fcId, byte level, Date lastReveiwed) {
        this.id = id;
        this.uId = uId;
        this.fcId = fcId;
        this.level = level;
        this.lastReveiwed = lastReveiwed;
    }

    @Ignore
    public UserCardLevel(String uId, String fcId, byte level, Date lastReveiwed) {
        this.uId = uId;
        this.fcId = fcId;
        this.level = level;
        this.lastReveiwed = lastReveiwed;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getUId() {
        return uId;
    }

    public void setUId(String uId) {
        this.uId = uId;
    }

    public String getFcId() {
        return fcId;
    }

    public void setFcId(String fcId) {
        this.fcId = fcId;
    }

    public byte getLevel() {
        return level;
    }

    public void setLevel(byte level) {
        this.level = level;
    }

    public Date getLastReveiwed() {
        return lastReveiwed;
    }

    public void setLastReveiwed(Date lastReveiwed) {
        this.lastReveiwed = lastReveiwed;
    }

    public static List<UserCardLevel> convertFlashCardsToUserCardLevels(List<FlashCard> cards, String uid) {
        byte level = 1;
        Date lastReviewed = new Date();
        List<UserCardLevel> ucl = new ArrayList<>();
        for (FlashCard fc : cards) {
            ucl.add(new UserCardLevel(uid, fc.getFcId(), level, lastReviewed));
        }
        return ucl;
    }

    public static UserCardLevel createUserCardLevel(String fcId, String uid) {
        byte level = 1;
        Date lastReviewed = new Date();
        UserCardLevel ucl = new UserCardLevel(uid, fcId, level, lastReviewed);
        return ucl;
    }

    public static List<String> getCardIdsFromList(List<UserCardLevel> list) {
        List<String> ids = new ArrayList<>();
        for (UserCardLevel ucl : list) {
            ids.add(ucl.getFcId());
        }
        return ids;
    }
}
