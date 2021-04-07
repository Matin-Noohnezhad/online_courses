package com.example.matin_noohnezhad_assignment_2.database;

import androidx.room.TypeConverter;

import com.example.matin_noohnezhad_assignment_2.task.TaskType;
import com.example.matin_noohnezhad_assignment_2.user.Gender;

import java.util.Date;

public class Converters {

    @TypeConverter
    public static Date fromTimestamp(Long value) {
        return value == null ? null : new Date(value);
    }

    @TypeConverter
    public static Long dateToTimestamp(Date date) {
        return date == null ? null : date.getTime();
    }

    @TypeConverter
    public static byte taskTypeToNum(TaskType taskType) {
        switch (taskType) {
            case JOB:
                return 1;
            case UNIVERSITY:
                return 2;
            case SHOPPING:
                return 3;
            case KHEDMAT:
                return 4;
            case EBADAT:
                return 5;
            case NEZAFAT:
                return 6;
        }
        return -1;
    }

    @TypeConverter
    public static byte genderToNum(Gender gender) {
        switch (gender) {
            case MALE:
                return 1;
            case FEMALE:
                return 2;
        }
        return -1;
    }

    @TypeConverter
    public static TaskType toTaskType(byte type) {
        switch (type) {
            case 1:
                return TaskType.JOB;
            case 2:
                return TaskType.UNIVERSITY;
            case 3:
                return TaskType.SHOPPING;
            case 4:
                return TaskType.KHEDMAT;
            case 5:
                return TaskType.EBADAT;
            case 6:
                return TaskType.NEZAFAT;
        }
        return null;
    }

    @TypeConverter
    public static Gender toGender(byte type) {
        switch (type) {
            case 1:
                return Gender.MALE;
            case 2:
                return Gender.FEMALE;
        }
        return null;
    }
}
