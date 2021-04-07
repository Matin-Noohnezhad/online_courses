package com.example.matin_noohnezhad_assignment_2.activities;

import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.User;

public class PageActivity extends AppCompatActivity {

    private static User currentUser;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_page);
    }

    public void dueTasksClicked(View view) {
        Context context = getApplicationContext();
//        CharSequence text = "Due tasks is not implemented yet!!!";
//        int duration = Toast.LENGTH_SHORT;
//        Toast toast = Toast.makeText(context, text, duration);
//        toast.show();
        Intent intent = new Intent(context, PageDueTasks.class);
        startActivity(intent);
    }

    public void exportTasksClicked(View view) {
        Context context = getApplicationContext();
        CharSequence text = "Export tasks is not implemented yet!!!";
        int duration = Toast.LENGTH_SHORT;
        Toast toast = Toast.makeText(context, text, duration);
        toast.show();
    }

    public void importTasksClicked(View view) {
        Context context = getApplicationContext();
        CharSequence text = "Import tasks is not implemented yet!!!";
        int duration = Toast.LENGTH_SHORT;
        Toast toast = Toast.makeText(context, text, duration);
        toast.show();
    }

    public void archivedTasksClicked(View view) {
        Context context = getApplicationContext();
//        CharSequence text = "Archived tasks is not implemented yet!!!";
//        int duration = Toast.LENGTH_SHORT;
//        Toast toast = Toast.makeText(context, text, duration);
//        toast.show();
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
}
