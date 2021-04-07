package com.example.flashcard_project.beans;

import androidx.annotation.NonNull;
import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

import com.google.gson.annotations.SerializedName;

@Entity
public class User {
    @PrimaryKey
    @NonNull
    @SerializedName("objectId")
    private String uid;
    @ColumnInfo
    private String name;
    @ColumnInfo
    private String password;
    @ColumnInfo
    private String email;
    @ColumnInfo(name = "user_token")
    @SerializedName("user-token")
    private String userToken;

    public User(String uid, String name, String password, String email, String userToken) {
        this.uid = uid;
        this.name = name;
        this.password = password;
        this.email = email;
        this.userToken = userToken;
    }

    public String getUid() {
        return uid;
    }

    public void setUid(String uid) {
        this.uid = uid;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getUserToken() {
        return userToken;
    }

    public void setUserToken(String userToken) {
        this.userToken = userToken;
    }

    @Override
    public String toString() {
        return "User{" +
                "uid='" + uid + '\'' +
                ", name='" + name + '\'' +
                ", password='" + password + '\'' +
                ", email='" + email + '\'' +
                ", userToken='" + userToken + '\'' +
                '}';
    }
}
