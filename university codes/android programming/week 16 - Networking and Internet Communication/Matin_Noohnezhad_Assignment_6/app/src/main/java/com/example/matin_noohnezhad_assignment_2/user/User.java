package com.example.matin_noohnezhad_assignment_2.user;

import androidx.annotation.NonNull;
import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

import com.google.gson.annotations.SerializedName;

@Entity
public class User {

    @PrimaryKey
    @NonNull
    @SerializedName("objectId")
    private String uid;
    @ColumnInfo
    @SerializedName("name")
    private String username;
    @ColumnInfo
    private String password;
    @ColumnInfo
    private String email;
    @ColumnInfo(name = "mobile_no")
    private String mobileNo;
    @ColumnInfo
    private String language;
    @ColumnInfo
    private Gender gender;
    @ColumnInfo(name = "send_email")
    private boolean sendEmail;
    @ColumnInfo(name = "send_sms")
    private boolean sendSms;
    @ColumnInfo(name = "user_token")
    @SerializedName("user-token")
    private String userToken;

    @Ignore
    public User() {

    }

    public User(String username, String password, String email, String mobileNo, String language, Gender gender, boolean sendEmail, boolean sendSms) {
        this.username = username;
        this.password = password;
        this.email = email;
        this.mobileNo = mobileNo;
        this.language = language;
        this.gender = gender;
        this.sendEmail = sendEmail;
        this.sendSms = sendSms;
    }

    public String getUid() {
        return uid;
    }

    public void setUid(String uid) {
        this.uid = uid;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
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

    public String getMobileNo() {
        return mobileNo;
    }

    public void setMobileNo(String mobileNo) {
        this.mobileNo = mobileNo;
    }

    public String getLanguage() {
        return language;
    }

    public void setLanguage(String language) {
        this.language = language;
    }

    public Gender getGender() {
        return gender;
    }

    public void setGender(Gender gender) {
        this.gender = gender;
    }

    public boolean isSendEmail() {
        return sendEmail;
    }

    public void setSendEmail(boolean sendEmail) {
        this.sendEmail = sendEmail;
    }

    public boolean isSendSms() {
        return sendSms;
    }

    public void setSendSms(boolean sendSms) {
        this.sendSms = sendSms;
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
                ", username='" + username + '\'' +
                ", password='" + password + '\'' +
                ", email='" + email + '\'' +
                ", mobileNo='" + mobileNo + '\'' +
                ", language='" + language + '\'' +
                ", gender=" + gender +
                ", sendEmail=" + sendEmail +
                ", sendSms=" + sendSms +
                ", userToken='" + userToken + '\'' +
                '}';
    }
}
