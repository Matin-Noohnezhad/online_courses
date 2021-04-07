package com.example.matin_noohnezhad_assignment_1;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class SecondActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
    }

    public void checkInputUrl(View view) {

        EditText urlInput = findViewById(R.id.url_input);
        String s = urlInput.getText().toString();
        if (s.equals("") || s.isEmpty()) {
            Context context = getApplicationContext();
            CharSequence text = "you must fill the url field!!!";
            int duration = Toast.LENGTH_SHORT;
            Toast toast = Toast.makeText(context, text, duration);
            toast.show();
        } else {
            Intent resultIntent = new Intent();
            resultIntent.putExtra("input_url", s);
            setResult(Activity.RESULT_OK, resultIntent);
            finish();
        }
    }
}
