package com.example.matin_noohnezhad_assignment_2.database.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import com.example.matin_noohnezhad_assignment_2.task.Task;

import java.util.List;

@Dao
public interface TaskDao {

    @Query("select * from Task where uid_fk=:id")
    List<Task> getAll(int id);

    @Insert
    long insert(Task task);

    @Query("delete from Task where tid=:id")
    void delete(int id);

    @Query("update Task set is_archived=:isArchived where tid=:id ")
    void isArchiveUpdate(int id, boolean isArchived);

}
