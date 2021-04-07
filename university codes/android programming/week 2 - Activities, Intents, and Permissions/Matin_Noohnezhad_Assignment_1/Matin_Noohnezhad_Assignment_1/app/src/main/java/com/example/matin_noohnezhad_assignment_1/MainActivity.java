package com.example.matin_noohnezhad_assignment_1;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    public static final int URL_REQUEST = 1;//request code

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void goToSecondActivity(View view) {
        Intent intent = new Intent(this, SecondActivity.class);
        startActivityForResult(intent, URL_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
//        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == URL_REQUEST) {
            if (resultCode == Activity.RESULT_OK) {
                String result = (String) data.getExtras().getString("input_url");
                TextView urlTextView = findViewById(R.id.url_result);
                urlTextView.setText(result);
                urlTextView.setVisibility(View.VISIBLE);
                findViewById(R.id.open_url_btn).setVisibility(View.VISIBLE);
            }
        }
    }

    public void openUrl(View view) {
        TextView urlTextView = findViewById(R.id.url_result);
        String url = urlTextView.getText().toString();
        Uri webpage = Uri.parse("https://" + url);
        Intent webIntent = new Intent(Intent.ACTION_VIEW, webpage);
        startActivity(webIntent);
    }
}
