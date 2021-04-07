package com.example.matin_noohnezhad_assignment_2.utils;

import android.util.Pair;

import com.example.matin_noohnezhad_assignment_2.Task;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

public class JsonConverter {

    private static Gson gson = new Gson();

    public static String toJson(List<Task> dueTasks, List<Task> archivedTasks) {
        Type type = new TypeToken<Pair<List<Task>, List<Task>>>() {
        }.getType();
        Pair<List<Task>, List<Task>> pair;
        pair = new Pair<>(dueTasks, archivedTasks);
        String json = gson.toJson(pair, type);
        return json;
    }

    public static Pair<List<Task>, List<Task>> jsonToPair(String json) {
        Type type = new TypeToken<Pair<List<Task>, List<Task>>>() {
        }.getType();
        Pair<List<Task>, List<Task>> pair;
        pair = gson.fromJson(json, type);
        return pair;
    }

}
