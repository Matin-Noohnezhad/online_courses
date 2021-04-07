package com.example.matin_noohnezhad_assignment_2.database.asyn_tasks.inputs;

import com.example.matin_noohnezhad_assignment_2.task.Task;

public class TaskOperationInput {

    private Task task;
    private String id;

    public TaskOperationInput(Task task, String id) {
        this.task = task;
        this.id = id;
    }

    public Task getTask() {
        return task;
    }

    public void setTask(Task task) {
        this.task = task;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }
}
