package com.example.matin_noohnezhad_assignment_2;

import android.content.Context;
import android.content.SharedPreferences;

import com.google.gson.Gson;

public class PreferenceManager {

    private static final String PREF_FILE_NAME = "com.example.matin_noohnezhad_assignment_2";

    private static final String PREF_KEY_NAME = "name";
    private static final String PREF_KEY_GRADE = "grade";

    private static PreferenceManager sInstance;
    private static Gson gson;
    private Context mContext;
    private SharedPreferences preferences;

    private PreferenceManager(Context context) {
        mContext = context;
        preferences = mContext.getSharedPreferences(PREF_FILE_NAME, Context.MODE_PRIVATE);
    }

    public static PreferenceManager getInstance(Context context) {
        if (sInstance == null) {
            sInstance = new PreferenceManager(context);
            gson = new Gson();
        }
        return sInstance;
    }

    public void putString(String key, String value) {
        preferences.edit().putString(key, value).apply();

    }

    public String getString(String key, String defaultValue) {
        return preferences.getString(key, defaultValue);
    }

    public void putUser(User user) {
        String json = gson.toJson(user);
        putString(user.getUsername(), json);
    }

    public User getUser(String key, String defaultValue) {
        String json = getString(key, defaultValue);
        if(json.equals(defaultValue)){
            return null;
        }
        User user = gson.fromJson(json, User.class);
        return user;
    }
}
