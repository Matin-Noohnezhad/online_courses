package com.example.matin_noohnezhad_assignment_2.activities;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.util.Pair;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.task.Task;
import com.example.matin_noohnezhad_assignment_2.user.User;
import com.example.matin_noohnezhad_assignment_2.file.FileAccessHelper;
import com.example.matin_noohnezhad_assignment_2.utils.JsonConverter;
import com.example.matin_noohnezhad_assignment_2.utils.TaskListHolder;

import java.util.List;

public class PageActivity extends AppCompatActivity {

    private static final int WRITE_FILE_PICKER_REQUEST_CODE = 1;
    private static final int WRITE_PERMISSION_REQUEST_CODE = 1;
    private static final int READ_FILE_PICKER_REQUEST_CODE = 2;
    private static final int READ_PERMISSION_REQUEST_CODE = 2;
    private static User currentUser;
    private FileAccessHelper fileAccessHelper = null;
    private Uri fileWriteUri = null;
    private Uri fileReadUri = null;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_page);
    }

    public void dueTasksClicked(View view) {
        Context context = getApplicationContext();
        Intent intent = new Intent(context, PageDueTasks.class);
        startActivity(intent);
    }


    public void archivedTasksClicked(View view) {
        Context context = getApplicationContext();
        Intent intent = new Intent(context, PageArchivedTasks.class);
        startActivity(intent);
    }

    public void profileClicked(View view) {
        // 1. Instantiate an <code><a href="/reference/android/app/AlertDialog.Builder.html">AlertDialog.Builder</a></code> with its constructor
        AlertDialog.Builder builder = new AlertDialog.Builder(this);

// 2. Chain together various setter methods to set the dialog characteristics
        builder.setMessage(getUserInfo())
                .setTitle(R.string.dialog_box_title)
                .setPositiveButton(R.string.ok, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        // do nothing
                    }
                });
        ;

// 3. Get the <code><a href="/reference/android/app/AlertDialog.html">AlertDialog</a></code> from <code><a href="/reference/android/app/AlertDialog.Builder.html#create()">create()</a></code>
        AlertDialog dialog = builder.create();
//        builder.show();
        dialog.show();
    }

    private String getUserInfo() {
        String info = "";
        info += "Username: " + currentUser.getUsername() + "\n";
        info += "Email Address: " + currentUser.getEmail() + "\n";
        info += "Mobile No: " + currentUser.getMobileNo() + "\n";
        info += "Language: " + currentUser.getLanguage() + "\n";
        info += "Gender : " + currentUser.getGender() + "\n";
        if (currentUser.isSendEmail())
            info += "Send Email : " + "YES" + "\n";
        else
            info += "Send Email : " + "NO" + "\n";
        if (currentUser.isSendSms())
            info += "Send SMS : " + "YES" + "\n";
        else
            info += "Send SMS : " + "NO" + "\n";
        return info;

    }

    public void exitClicked(View view) {
        Intent intent = new Intent(Intent.ACTION_MAIN);
        intent.addCategory(Intent.CATEGORY_HOME);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
//        System.exit(0);
    }

    public static User getCurrentUser() {
        return currentUser;
    }

    public static void setCurrentUser(User currentUser) {
        PageActivity.currentUser = currentUser;
    }

    public void exportTasksClicked(View view) {
        Context context = getApplicationContext();
        fileAccessHelper = FileAccessHelper.getInstance(context);
        openFilePickerforWrite();
    }

    public void importTasksClicked(View view) {
        Context context = getApplicationContext();
        fileAccessHelper = FileAccessHelper.getInstance(context);
        openFilePickerforRead();
    }

    private void openFilePickerforWrite() {
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.setType("text/plain");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        intent.putExtra(Intent.EXTRA_TITLE, "test");
        startActivityForResult(intent, WRITE_FILE_PICKER_REQUEST_CODE);
    }

    private void openFilePickerforRead() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("text/plain");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        startActivityForResult(intent, READ_FILE_PICKER_REQUEST_CODE);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if ((requestCode != WRITE_FILE_PICKER_REQUEST_CODE) || (resultCode != RESULT_OK)) {
            if ((requestCode != READ_FILE_PICKER_REQUEST_CODE) || (resultCode != RESULT_OK)) {
                Toast.makeText(getApplicationContext(), getString(R.string.file_path_not_provided_error), Toast.LENGTH_LONG).show();
                return;
            }

            fileReadUri = (data != null) ? data.getData() : null;
            if (fileReadUri == null) {
                Toast.makeText(getApplicationContext(), getString(R.string.file_path_not_provided_error), Toast.LENGTH_LONG).show();
                return;
            }

            checkReadPermission();
            return;
        }

        fileWriteUri = (data != null) ? data.getData() : null;
        if (fileWriteUri == null) {
            Toast.makeText(getApplicationContext(), getString(R.string.file_path_not_provided_error), Toast.LENGTH_LONG).show();
            return;
        }
        checkWritePermission();
    }

    private void checkReadPermission() {
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.READ_EXTERNAL_STORAGE) ==
                PackageManager.PERMISSION_GRANTED) {
            readFileProcess();
        } else {
            ActivityCompat.requestPermissions(PageActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    READ_PERMISSION_REQUEST_CODE);
        }
    }

    private void checkWritePermission() {
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE) ==
                PackageManager.PERMISSION_GRANTED) {
            writeFileProcess();
        } else {
            ActivityCompat.requestPermissions(PageActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    WRITE_PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode != WRITE_PERMISSION_REQUEST_CODE) {
            if (requestCode != READ_PERMISSION_REQUEST_CODE) {
                return;
            }

            if ((grantResults.length > 0) && (grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                readFileProcess();
            } else {
                Toast.makeText(getApplicationContext(), R.string.read_permission_grant_error, Toast.LENGTH_LONG).show();
            }
            return;
        }

        if ((grantResults.length > 0) && (grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
            writeFileProcess();
        } else {
            Toast.makeText(getApplicationContext(), R.string.write_permission_grant_error, Toast.LENGTH_LONG).show();
        }
    }

    private void writeFileProcess() {
        if (fileWriteUri == null) {
            Toast.makeText(getApplicationContext(), R.string.file_path_not_provided_error, Toast.LENGTH_LONG).show();
            return;
        }

        String content = JsonConverter.toJson(TaskListHolder.getInstance().getDueTaskList(), TaskListHolder.getInstance().getArchivedTaskList());
        int errorCode = fileAccessHelper.writeToTheFile(fileWriteUri, content);
        if (errorCode > 0) {
            String errorMessage = fileAccessHelper.getFileErrorMessage(errorCode, getApplicationContext());
            Toast.makeText(getApplicationContext(), errorMessage, Toast.LENGTH_LONG).show();
            return;
        }

        Toast.makeText(getApplicationContext(), getString(R.string.writing_file_successful_message), Toast.LENGTH_LONG).show();
    }

    private void readFileProcess() {
        String filePath = (fileReadUri != null) ? fileReadUri.getPath() : null;
        if (fileReadUri == null) {
            Toast.makeText(getApplicationContext(), R.string.file_path_not_provided_error, Toast.LENGTH_LONG).show();
            return;
        }

        Pair<String, Integer> result = fileAccessHelper.readTheFile(fileReadUri);
        String fileContents = result.first;
        int errorCode = result.second;
        if (fileContents == null) {
            String errorMessage = fileAccessHelper.getFileErrorMessage(errorCode, PageActivity.this);
            Toast.makeText(getApplicationContext(), errorMessage, Toast.LENGTH_LONG).show();
            return;
        }

        Pair<List<Task>, List<Task>> p = JsonConverter.jsonToPair(fileContents);
        List<Task> dueTasks = p.first;
        List<Task> archivedTasks = p.second;
        TaskListHolder.getInstance().concatToDueTask(dueTasks);
        TaskListHolder.getInstance().concatToArchivedTask(archivedTasks);

        Toast.makeText(getApplicationContext(), R.string.reading_file_successful_message, Toast.LENGTH_LONG).show();
    }

}
