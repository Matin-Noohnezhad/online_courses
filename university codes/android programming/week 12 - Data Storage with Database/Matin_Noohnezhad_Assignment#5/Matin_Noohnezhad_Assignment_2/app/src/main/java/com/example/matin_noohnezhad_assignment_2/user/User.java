package com.example.matin_noohnezhad_assignment_2.user;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

@Entity
public class User {

    @PrimaryKey(autoGenerate = true)
    private int uid;
    @ColumnInfo
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

    @Ignore
    public User(){

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

    public int getUid() {
        return uid;
    }

    public void setUid(int uid) {
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
}
