package com.example.matin_noohnezhad_assignment_2.file;

import android.content.Context;
import android.net.Uri;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.util.Pair;

import com.example.matin_noohnezhad_assignment_2.R;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class FileAccessHelper {


    public static final int EXTERNAL_STORAGE_NOT_READY_ERROR_CODE = 1;
    public static final int FILE_DOES_NOT_EXIST_ERROR_CODE = 2;
    public static final int FILE_READ_ERROR_CODE = 3;
    public static final int FILE_WRITE_ERROR_CODE = 4;
    public static final int FILE_PATH_NOT_PROVIDED = 5;

    private static FileAccessHelper instance = null;

    private Context context = null;

    private FileAccessHelper(Context c) {
        this.context = c;
    }

    public static FileAccessHelper getInstance(Context c) {
        if (instance == null) {
            instance = new FileAccessHelper(c);
        }
        return instance;
    }


    private boolean isExternalStorageAvailable() {
        return (Environment.getExternalStorageState().equalsIgnoreCase(Environment.MEDIA_MOUNTED));
    }

    public Pair<String, Integer> readTheFile(Uri fileUri) {
        if (!isExternalStorageAvailable()) {
            return new Pair<String, Integer>(null, EXTERNAL_STORAGE_NOT_READY_ERROR_CODE);
        }

        String wholeText = "";

        try {
            InputStream inputStream = context.getContentResolver().openInputStream(fileUri);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"));

            String line = "";
            while ((line = reader.readLine()) != null) {
                wholeText = wholeText + line + "\n";
            }

            reader.close();
            inputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return new Pair<String, Integer>(null, FILE_DOES_NOT_EXIST_ERROR_CODE);
        } catch (IOException e) {
            e.printStackTrace();
            return new Pair<String, Integer>(null, FILE_READ_ERROR_CODE);
        }

        return new Pair<String, Integer>(wholeText, 0);
    }

    public int writeToTheFile(Uri fileUri, String contents) {
        if (!isExternalStorageAvailable()) {
            return EXTERNAL_STORAGE_NOT_READY_ERROR_CODE;
        }

        try {
            ParcelFileDescriptor pfd = context.getContentResolver().openFileDescriptor(fileUri, "w");
            FileOutputStream outStream = new FileOutputStream(pfd.getFileDescriptor());
            outStream.write(contents.getBytes("UTF-8"));

            outStream.close();
            pfd.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return FILE_DOES_NOT_EXIST_ERROR_CODE;
        } catch (IOException e) {
            e.printStackTrace();
            return FILE_WRITE_ERROR_CODE;
        }

        return 0;
    }

    public String getFileErrorMessage(int errorCode, Context context) {
        String errorMessage = "";
        switch (errorCode) {
            case EXTERNAL_STORAGE_NOT_READY_ERROR_CODE:
                errorMessage = context.getString(R.string.external_storage_not_ready_error);
                break;
            case FILE_DOES_NOT_EXIST_ERROR_CODE:
                errorMessage = context.getString(R.string.file_does_not_exist_error);
                break;
            case FILE_READ_ERROR_CODE:
                errorMessage = context.getString(R.string.reading_file_error);
                break;
            case FILE_WRITE_ERROR_CODE:
                errorMessage = context.getString(R.string.writing_file_error);
                break;
            case FILE_PATH_NOT_PROVIDED:
                errorMessage = context.getString(R.string.file_path_not_provided_error);
                break;
        }
        return errorMessage;
    }
}
