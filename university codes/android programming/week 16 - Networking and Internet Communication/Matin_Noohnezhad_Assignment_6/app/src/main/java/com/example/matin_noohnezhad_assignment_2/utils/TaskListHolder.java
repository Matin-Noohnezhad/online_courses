package com.example.matin_noohnezhad_assignment_2.utils;

import com.example.matin_noohnezhad_assignment_2.task.Task;

import java.util.ArrayList;
import java.util.List;

public class TaskListHolder {

    private static TaskListHolder instance;
    private List<Task> dueTaskList;
    private List<Task> archivedTaskList;


    public static TaskListHolder getInstance() {
        if (instance == null) {
            instance = new TaskListHolder();
            instance.dueTaskList = new ArrayList<>();
            instance.archivedTaskList = new ArrayList<>();
        }
        return instance;
    }

    public void setDueTaskList(List<Task> dueTaskList) {
        this.dueTaskList = dueTaskList;
    }

    public void setArchivedTaskList(List<Task> archivedTaskList) {
        this.archivedTaskList = archivedTaskList;
    }

    public List<Task> getDueTaskList() {
        return dueTaskList;
    }

    public List<Task> getArchivedTaskList() {
        return archivedTaskList;
    }

    public void addToDueTasks(Task t) {
        dueTaskList.add(t);
    }

    public void addToArchivedTasks(Task t) {
        archivedTaskList.add(t);
    }

    public void concatToDueTask(List<Task> tasks) {
        dueTaskList.addAll(tasks);
    }

    public void concatToArchivedTask(List<Task> tasks) {
        archivedTaskList.addAll(tasks);
    }

}
