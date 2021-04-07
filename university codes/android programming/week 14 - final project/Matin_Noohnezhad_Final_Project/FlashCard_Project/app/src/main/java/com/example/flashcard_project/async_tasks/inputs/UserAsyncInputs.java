package com.example.flashcard_project.async_tasks.inputs;

import com.example.flashcard_project.beans.User;

import java.util.List;

public class UserAsyncInputs {

    private List<User> users;
    private User user;
    private String uid;

    public UserAsyncInputs(List<User> users, User user, String uid) {
        this.users = users;
        this.user = user;
        this.uid = uid;
    }

    public List<User> getUsers() {
        return users;
    }

    public void setUsers(List<User> users) {
        this.users = users;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public String getUid() {
        return uid;
    }

    public void setUid(String uid) {
        this.uid = uid;
    }
}
