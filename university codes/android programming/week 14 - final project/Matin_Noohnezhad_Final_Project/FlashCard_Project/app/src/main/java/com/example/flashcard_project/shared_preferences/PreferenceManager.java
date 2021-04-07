package com.example.flashcard_project.shared_preferences;

import android.content.Context;
import android.content.SharedPreferences;

public class PreferenceManager {

    private static final String PREF_FILE_NAME = "com.example.flash_card_project";

    private static final String PREF_KEY_CURRENT_USER_ID = "current_user_id";

    private static final String NO_USER = "NULL";

    private static PreferenceManager sInstance;
    private Context mContext;
    private SharedPreferences preferences;

    private PreferenceManager(Context context) {
        mContext = context;
        preferences = mContext.getSharedPreferences(PREF_FILE_NAME, Context.MODE_PRIVATE);
    }

    public static PreferenceManager getInstance(Context context) {
        if (sInstance == null)
            sInstance = new PreferenceManager(context);
        return sInstance;
    }

    private void putString(String key, String value) {
        preferences.edit().putString(key, value).apply();

    }

    private String getString(String key, String defaultValue) {
        return preferences.getString(key, defaultValue);
    }

    public void putCurrentUserId(String id) {
        if (id == null)
            putString(PREF_KEY_CURRENT_USER_ID, NO_USER);
        else
            putString(PREF_KEY_CURRENT_USER_ID, id);
    }

    public String getCurrentUserId() {
        String id = null;
        String defaultValue = NO_USER;
        String savedData = getString(PREF_KEY_CURRENT_USER_ID, defaultValue);
        if (!savedData.equals(defaultValue)) {
            id = savedData;
        }
        return id;
    }
}
