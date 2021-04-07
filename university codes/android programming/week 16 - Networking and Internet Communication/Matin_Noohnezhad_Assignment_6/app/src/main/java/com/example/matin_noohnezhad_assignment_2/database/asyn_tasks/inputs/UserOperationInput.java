package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs;

import com.example.matin_noohnezhad_assignment_2.user.User;

public class UserOperationInput {

    private User user;
    private String id;

    public UserOperationInput(User user, String id) {
        this.user = user;
        this.id = id;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }
}
