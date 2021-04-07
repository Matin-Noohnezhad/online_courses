package com.example.flashcard_project.beans;

import androidx.annotation.NonNull;
import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

import com.google.gson.annotations.SerializedName;

import java.util.Date;

@Entity
public class FlashCard {

    @PrimaryKey
    @NonNull
    @SerializedName("objectId")
    private String fcId;
    @ColumnInfo
    private String word;
    @ColumnInfo
    private String definition;
    @ColumnInfo(name = "word_usage")
    private String wordUsage;

    public FlashCard(String fcId, String word, String definition, String wordUsage) {
        this.fcId = fcId;
        this.word = word;
        this.definition = definition;
        this.wordUsage = wordUsage;
    }

    @Ignore
    public FlashCard(String word, String definition, String wordUsage) {
        this.word = word;
        this.definition = definition;
        this.wordUsage = wordUsage;
    }


    public String getFcId() {
        return fcId;
    }

    public void setFcId(String fcId) {
        this.fcId = fcId;
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public String getDefinition() {
        return definition;
    }

    public void setDefinition(String definition) {
        this.definition = definition;
    }

    public String getWordUsage() {
        return wordUsage;
    }

    public void setWordUsage(String wordUsage) {
        this.wordUsage = wordUsage;
    }

    @Override
    public String toString() {
        return "FlashCard{" +
                "fcId='" + fcId + '\'' +
                ", word='" + word + '\'' +
                ", definition='" + definition + '\'' +
                ", wordUsage='" + wordUsage + '\'' +
                '}';
    }
}
