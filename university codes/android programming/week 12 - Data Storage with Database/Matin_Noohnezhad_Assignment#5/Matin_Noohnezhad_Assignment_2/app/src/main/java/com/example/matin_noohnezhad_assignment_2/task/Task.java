package com.example.matin_noohnezhad_assignment_2.task;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.PrimaryKey;

import java.util.Date;

@Entity
public class Task {

    @PrimaryKey(autoGenerate = true)
    private int tid;
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
    private int uidFk;

    public Task(TaskType type, String title, String description, Date dueDate, String dueHour, String dueMinute, int uidFk) {
        this.type = type;
        this.title = title;
        this.description = description;
        this.dueDate = dueDate;
        this.dueHour = dueHour;
        this.dueMinute = dueMinute;
        this.isArchived = false;
        this.uidFk = uidFk;
    }

    public int getUidFk() {
        return uidFk;
    }

    public void setUidFk(int uidFk) {
        this.uidFk = uidFk;
    }

    public int getTid() {
        return tid;
    }

    public void setTid(int tid) {
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
}
