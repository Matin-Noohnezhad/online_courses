package com.example.matin_noohnezhad_assignment_2.task;

import androidx.annotation.NonNull;
import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

import com.google.gson.annotations.SerializedName;

import java.util.Date;

@Entity
public class Task {

    @PrimaryKey
    @NonNull
    @SerializedName("objectId")
    private String tid;
    @ColumnInfo
    private TaskType type;
    @ColumnInfo
    private String title;
    @ColumnInfo
    private String description;
    @ColumnInfo(name = "due_date")
    private Date dueDate;
    @ColumnInfo(name = "due_hour")
    private String dueHour;
    @ColumnInfo(name = "due_minute")
    private String dueMinute;
    @ColumnInfo(name = "is_archived")
    private boolean isArchived;
    @ColumnInfo(name = "uid_fk")
    @SerializedName("ownerId")
    private String uidFk;

    public Task(TaskType type, String title, String description, Date dueDate, String dueHour, String dueMinute, String uidFk) {
        this.type = type;
        this.title = title;
        this.description = description;
        this.dueDate = dueDate;
        this.dueHour = dueHour;
        this.dueMinute = dueMinute;
        this.isArchived = false;
        this.uidFk = uidFk;
    }

    public String getUidFk() {
        return uidFk;
    }

    public void setUidFk(String uidFk) {
        this.uidFk = uidFk;
    }

    public String getTid() {
        return tid;
    }

    public void setTid(String tid) {
        this.tid = tid;
    }

    public TaskType getType() {
        return type;
    }

    public void setType(TaskType type) {
        this.type = type;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Date getDueDate() {
        return dueDate;
    }

    public void setDueDate(Date dueDate) {
        this.dueDate = dueDate;
    }

    public String getDueHour() {
        return dueHour;
    }

    public void setDueHour(String dueHour) {
        this.dueHour = dueHour;
    }

    public String getDueMinute() {
        return dueMinute;
    }

    public void setDueMinute(String dueMinute) {
        this.dueMinute = dueMinute;
    }

    public boolean isArchived() {
        return isArchived;
    }

    public void setArchived(boolean archived) {
        isArchived = archived;
    }

    @Override
    public String toString() {
        return "Task{" +
                "tid='" + tid + '\'' +
                ", type=" + type +
                ", title='" + title + '\'' +
                ", description='" + description + '\'' +
                ", dueDate=" + dueDate +
                ", dueHour='" + dueHour + '\'' +
                ", dueMinute='" + dueMinute + '\'' +
                ", isArchived=" + isArchived +
                ", uidFk='" + uidFk + '\'' +
                '}';
    }
}
