package com.example.matin_noohnezhad_assignment_2;

import com.example.matin_noohnezhad_assignment_2.user.User;

public class ValidationAnswer {

    private boolean isValid;
    private User user;
    private String toastMessage;

    public ValidationAnswer(boolean isValid, User user, String toastMessage) {
        this.isValid = isValid;
        this.user = user;
        this.toastMessage = toastMessage;
    }

    public boolean isValid() {
        return isValid;
    }

    public void setValid(boolean valid) {
        isValid = valid;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public String getToastMessage() {
        return toastMessage;
    }

    public void setToastMessage(String toastMessage) {
        this.toastMessage = toastMessage;
    }
}
