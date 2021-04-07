package com.example.matin_noohnezhad_assignment_2.task;

import com.example.matin_noohnezhad_assignment_2.R;

public enum TaskType {
    JOB, UNIVERSITY, SHOPPING, KHEDMAT, EBADAT, NEZAFAT;

    public static int findIconByTaskType(TaskType tt) {
        int src = 0;
        switch (tt) {
            case JOB:
                src = R.drawable.job;
                break;
            case UNIVERSITY:
                src = R.drawable.university;
                break;
            case SHOPPING:
                src = R.drawable.shopping;
                break;
            case KHEDMAT:
                src = R.drawable.khedmat;
                break;
            case EBADAT:
                src = R.drawable.ebadat;
                break;
            case NEZAFAT:
                src = R.drawable.nezafat;
                break;
        }
        return src;
    }

    public static TaskType findTypeByString(String type) {
        TaskType taskType = null;
        switch (type) {
            case "JOB":
                taskType = taskType.JOB;
                break;
            case "UNIVERSITY":
                taskType = taskType.UNIVERSITY;
                break;
            case "SHOPPING":
                taskType = taskType.SHOPPING;
                break;
            case "KHEDMAT":
                taskType = taskType.KHEDMAT;
                break;
            case "EBADAT":
                taskType = taskType.EBADAT;
                break;
            case "NEZAFAT":
                taskType = taskType.NEZAFAT;
                break;
        }
        return taskType;
    }
}
