package com.example.matin_noohnezhad_assignment_2;

import java.util.ArrayList;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.List;

public class Task {

    private TaskType type;
    private String title;
    private String description;
    private Date dueDate;
    private String dueHour;
    private String dueMinute;
    private static List<Task> dueTaskList = generateDueTaskList();
    private static List<Task> archivedTaskList = generateArchivedTaskList();

    public Task(TaskType type, String title, String description, Date dueDate, String dueHour, String dueMinute) {
        this.type = type;
        this.title = title;
        this.description = description;
        this.dueDate = dueDate;
        this.dueHour = dueHour;
        this.dueMinute = dueMinute;
    }

    public static List<Task> getDueTaskList() {
        return dueTaskList;
    }

    public static List<Task> getArchivedTaskList() {
        return archivedTaskList;
    }

    public static void addToDueTasks(Task t) {
        dueTaskList.add(t);
    }

    public static void addToArchivedTasks(Task t) {
        archivedTaskList.add(t);
    }

    public TaskType getType() {
        return type;
    }

    public String getTitle() {
        return title;
    }

    public String getDescription() {
        return description;
    }

    public Date getDueDate() {
        return dueDate;
    }

    public String getDueHour() {
        return dueHour;
    }

    public String getDueMinute() {
        return dueMinute;
    }

    private static List<Task> generateDueTaskList() {
        List<Task> tasks = new ArrayList<>();
        tasks.add(new Task(TaskType.UNIVERSITY, "doing homework", "do android programming assignment 3", new GregorianCalendar(2018, 12 - 1, 23).getTime(), "23", "55"));
        tasks.add(new Task(TaskType.KHEDMAT, "kelas roohi", "kelas roohi ro sherkat mikonim", new GregorianCalendar(2016, 5 - 1, 13).getTime(), "19", "00"));
        return tasks;
    }

    private static List<Task> generateArchivedTaskList() {
        List<Task> tasks = new ArrayList<>();
        tasks.add(new Task(TaskType.NEZAFAT, "gereftan nakhoon", "nakhoonamonoo kutah konim", new GregorianCalendar(2017, 8 - 1, 11).getTime(), "8", "00"));
        tasks.add(new Task(TaskType.SHOPPING, "buying fruits", "chand ta mive bekharim", new GregorianCalendar(2015, 9 - 1, 9).getTime(), "15", "00"));
        return tasks;
    }
}
