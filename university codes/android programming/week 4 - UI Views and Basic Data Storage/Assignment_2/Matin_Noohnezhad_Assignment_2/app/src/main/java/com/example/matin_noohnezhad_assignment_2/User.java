package com.example.matin_noohnezhad_assignment_2;

public class User {

    private String username;
    private String password;
    private String email;
    private String mobileNo;
    private String language;
    private String gender;
    private boolean sendEmail;
    private boolean sendSms;


    public User() {
    }

    public User(String username, String password, String email, String mobileNo, String language, String gender, boolean sendEmail, boolean sendSms) {
        this.username = username;
        this.password = password;
        this.email = email;
        this.mobileNo = mobileNo;
        this.language = language;
        this.gender = gender;
        this.sendEmail = sendEmail;
        this.sendSms = sendSms;
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

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
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
