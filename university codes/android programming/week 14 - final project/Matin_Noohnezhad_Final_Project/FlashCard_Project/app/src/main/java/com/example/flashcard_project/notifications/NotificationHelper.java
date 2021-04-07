package com.example.flashcard_project.notifications;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;
import androidx.core.app.NotificationCompat;

import com.example.flashcard_project.R;

import java.security.PrivilegedAction;

public class NotificationHelper {

    private static final String CHANNEL_ID = "channel";
    private static final String CHANNEL_NAME = "Reminder";
    public static final int NOTIFICATION_ID = 1;
    private NotificationManager notificationManager;
    private static NotificationHelper instance;

    public static synchronized NotificationHelper getInstance(Context context) {
        if (instance == null) {
            instance = new NotificationHelper(context);
        }
        return instance;
    }

    private NotificationHelper(Context context) {
        notificationManager = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            createChannels();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    private void createChannels() {
        NotificationChannel channel = new NotificationChannel(CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_HIGH);
        channel.setDescription("To use send reminder notification");

        notificationManager.createNotificationChannel(channel);
    }

    public NotificationCompat.Builder createNotificationBuilder(String title, String message, PendingIntent pendingIntent, Context context) {
        NotificationCompat.Builder builder;

        builder = new NotificationCompat.Builder(context, CHANNEL_ID);
        builder.setSmallIcon(R.drawable.ic_launcher);
        builder.setContentTitle(title);
        builder.setContentText(message);
        builder.setAutoCancel(true);
        builder.setPriority(NotificationCompat.PRIORITY_MAX);
        builder.setCategory(NotificationCompat.CATEGORY_REMINDER);

        if (pendingIntent != null)
            builder.setContentIntent(pendingIntent);

        return builder;
    }

    public void makeNotification(NotificationCompat.Builder builder, int notificationId) {
        notificationManager.notify(notificationId, builder.build());
    }

}
