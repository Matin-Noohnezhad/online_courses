package com.example.flashcard_project.notifications;

import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import androidx.core.app.NotificationCompat;

import com.example.flashcard_project.activities.MainActivity;

public class AlarmBroadcastReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        //notification code
        Intent openAppIntent = new Intent(context, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(context, 0, openAppIntent, 0);
        NotificationHelper notificationHelper = NotificationHelper.getInstance(context);
        NotificationCompat.Builder builder = notificationHelper.createNotificationBuilder("Flash Card App",
                "Review your cards. It's been almost a day you don't open your app",
                pendingIntent,
                context
        );
        notificationHelper.makeNotification(builder, notificationHelper.NOTIFICATION_ID);
        //
    }
}
