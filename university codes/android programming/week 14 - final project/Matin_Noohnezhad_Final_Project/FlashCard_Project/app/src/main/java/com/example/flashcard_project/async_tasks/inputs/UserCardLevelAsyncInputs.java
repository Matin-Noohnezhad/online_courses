package com.example.flashcard_project.async_tasks.inputs;

import com.example.flashcard_project.beans.UserCardLevel;

import java.util.List;

public class UserCardLevelAsyncInputs {

    private List<UserCardLevel> items;
    private UserCardLevel item;
    private byte level;
    private String uid;

    public UserCardLevelAsyncInputs(List<UserCardLevel> items, UserCardLevel item) {
        this.items = items;
        this.item = item;
    }

    public UserCardLevelAsyncInputs(byte level, String uid) {
        this.level = level;
        this.uid = uid;
    }

    public List<UserCardLevel> getItems() {
        return items;
    }

    public void setItems(List<UserCardLevel> items) {
        this.items = items;
    }

    public UserCardLevel getItem() {
        return item;
    }

    public void setItem(UserCardLevel item) {
        this.item = item;
    }

    public byte getLevel() {
        return level;
    }

    public void setLevel(byte level) {
        this.level = level;
    }

    public String getUid() {
        return uid;
    }

    public void setUid(String uid) {
        this.uid = uid;
    }
}

