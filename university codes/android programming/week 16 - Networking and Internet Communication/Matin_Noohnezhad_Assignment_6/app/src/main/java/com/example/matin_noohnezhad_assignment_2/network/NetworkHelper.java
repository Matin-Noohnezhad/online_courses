package com.example.matin_noohnezhad_assignment_2.network;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.util.Log;

import com.android.volley.AuthFailureError;
import com.android.volley.NetworkResponse;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.example.matin_noohnezhad_assignment_2.R;
import com.example.matin_noohnezhad_assignment_2.task.Task;
import com.example.matin_noohnezhad_assignment_2.user.User;
import com.example.matin_noohnezhad_assignment_2.utils.Action;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeListener;
import com.example.matin_noohnezhad_assignment_2.utils.ItemChangeResult;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NetworkHelper {

    private static final String TAG = "NetworkHelper";
    private static NetworkHelper instance = null;
    private Context context;
    private RequestQueue requestQueue;
    private Gson gson = new Gson();
    private String hostUrl;
    private String appId;
    private String apiKey;

    private NetworkHelper(Context c) {
        context = c;
        hostUrl = context.getString(R.string.hostUrl);
        appId = context.getString(R.string.appId);
        apiKey = context.getString(R.string.apiKey);
        requestQueue = Volley.newRequestQueue(context);
    }

    public static NetworkHelper getInstance(Context c) {
        if (instance == null) {
            instance = new NetworkHelper(c);
        }
        return instance;
    }


    public boolean isNetworkConnected() {
        ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo net = cm.getActiveNetworkInfo();
        return ((net != null) && net.isConnected());
    }


    public void userSignIn(final String login, final String password, final ItemChangeListener<User> listener) {
        Error error = null;
        if (!isNetworkConnected()) {
            reportNetworkError(Action.SIGN_IN_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/users/login";

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "sign in response: " + response);
                if (response == null) {
                    reportNetworkError(Action.SIGN_IN_NET, R.string.network_sign_in_error, listener);
                    return;
                }

                User signedInUser = null;
                try {
                    signedInUser = gson.fromJson(response, new TypeToken<User>() {
                    }.getType());
                } catch (Exception ex) {
                    reportNetworkError(Action.SIGN_IN_NET, R.string.network_json_error, listener);
                    return;
                }

                Log.d(TAG, "sign in user received: " + signedInUser.toString());
                listener.itemChanged(new ItemChangeResult<User>(null, signedInUser, Action.SIGN_IN_NET, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.SIGN_IN_NET, R.string.network_sign_in_error, listener);
            }
        };

        StringRequest request = new StringRequest(Request.Method.POST, url, responseListener, errorListener) {
            @Override
            public byte[] getBody() throws AuthFailureError {
                String jsonStr = "{\"login\":\"" + login + "\",\"password\":\"" + password + "\"}";
                try {
                    return jsonStr.getBytes("utf-8");
                } catch (Exception ex) {
                    ex.printStackTrace();
                    return null;
                }
            }

            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<>();
                headers.put("Content-Type", "application/json");
                return headers;
            }
        };

        requestQueue.add(request);
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    public void userSignUp(User user, final ItemChangeListener<User> listener) {
        Error error = null;
        if (!isNetworkConnected()) {
            reportNetworkError(Action.SIGN_UP_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/users/register";

        String usrStr = null;
        try {
            usrStr = gson.toJson(user);
        } catch (Exception ex) {
            reportNetworkError(Action.SIGN_UP_NET, R.string.network_json_error, listener);
            return;
        }

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "sign up response: " + response);
                if (response == null) {
                    reportNetworkError(Action.SIGN_UP_NET, R.string.network_sign_up_error, listener);
                    return;
                }

                User signedUpUser = null;
                try {
                    signedUpUser = gson.fromJson(response, new TypeToken<User>() {
                    }.getType());
                } catch (Exception ex) {
                    reportNetworkError(Action.SIGN_UP_NET, R.string.network_json_error, listener);
                    return;
                }

                Log.d(TAG, "sign up user received: " + signedUpUser.toString());
                listener.itemChanged(new ItemChangeResult<User>(null, signedUpUser, Action.SIGN_UP_NET, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.SIGN_UP_NET, R.string.network_sign_up_error, listener);
            }
        };
        final String jsonStr = usrStr;
        StringRequest request = new StringRequest(Request.Method.POST, url, responseListener, errorListener) {
            @Override
            public byte[] getBody() throws AuthFailureError {
                try {
                    return jsonStr.getBytes("utf-8");
                } catch (Exception ex) {
                    ex.printStackTrace();
                    return null;
                }
            }

            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<>();
                headers.put("Content-Type", "application/json");
                return headers;
            }
        };

        requestQueue.add(request);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    public void checkUserTokenValidity(String userToken, final ItemChangeListener<Boolean> listener) {
        Error error = null;
        if (!isNetworkConnected()) {
            reportNetworkError(Action.CHECK_USER_TOKEN_VALIDITY, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/users/isvalidusertoken/" + userToken;

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "user-token validity response: " + response);
                if (response == null) {
                    reportNetworkError(Action.CHECK_USER_TOKEN_VALIDITY, R.string.network_check_user_token_validity_error, listener);
                    return;
                }

                Log.d(TAG, "user-token validity received: " + response);
                listener.itemChanged(new ItemChangeResult<Boolean>(null, Boolean.parseBoolean(response), Action.CHECK_USER_TOKEN_VALIDITY, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.CHECK_USER_TOKEN_VALIDITY, R.string.network_check_user_token_validity_error, listener);
            }
        };
        StringRequest request = new StringRequest(Request.Method.GET, url, responseListener, errorListener) {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<>();
                headers.put("Content-Type", "application/json");
                return headers;
            }
        };

        requestQueue.add(request);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    public void addNewTask(Task task, String userToken, final ItemChangeListener<Task> listener) {
        Error error = null;
        if (!isNetworkConnected()) {
            reportNetworkError(Action.INSERT_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/data/task";

        String taskStr = null;
        try {
            taskStr = gson.toJson(task);
        } catch (Exception ex) {
            reportNetworkError(Action.INSERT_NET, R.string.network_json_error, listener);
            return;
        }

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "insert task response: " + response);
                if (response == null) {
                    reportNetworkError(Action.INSERT_NET, R.string.network_add_new_task_error, listener);
                    return;
                }

                Task insertedTask = null;
                try {
                    insertedTask = gson.fromJson(response, new TypeToken<Task>() {
                    }.getType());
                } catch (Exception ex) {
                    reportNetworkError(Action.INSERT_NET, R.string.network_json_error, listener);
                    return;
                }

                Log.d(TAG, "insert new task received: " + insertedTask.toString());
                listener.itemChanged(new ItemChangeResult<Task>(null, insertedTask, Action.INSERT_NET, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.INSERT_NET, R.string.network_add_new_task_error, listener);
            }
        };

        final String jsonStr = taskStr;
        Log.d(TAG, "Inserting Json " + jsonStr + " using url " + url);
        StringRequest request = new StringRequest(Request.Method.POST, url, responseListener, errorListener) {
            @Override
            public byte[] getBody() throws AuthFailureError {
                try {
                    return jsonStr.getBytes("utf-8");
                } catch (Exception ex) {
                    ex.printStackTrace();
                    return null;
                }
            }

            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<>();
                headers.put("Content-Type", "application/json");
                return headers;
            }
        };

        requestQueue.add(request);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    public void removeTask(final Task task, final String userToken, final ItemChangeListener<Task> listener) {
        Error error = null;
        if (!isNetworkConnected()) {
            reportNetworkError(Action.DELETE_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/data/task/" + task.getTid();

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "delete task response: " + response);
                if (response == null) {
                    reportNetworkError(Action.DELETE_NET, R.string.network_remove_task_error, listener);
                    return;
                }

                Log.d(TAG, "delete task complet: " + task.toString());
                listener.itemChanged(new ItemChangeResult<Task>(null, task, Action.DELETE_NET, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.DELETE_NET, R.string.network_remove_task_error, listener);
            }
        };

        StringRequest request = new StringRequest(Request.Method.DELETE, url, responseListener, errorListener) {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<>();
                headers.put("Content-Type", "application/json");
                headers.put("user-token", userToken);
                return headers;
            }
        };

        requestQueue.add(request);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    public void updateIsArchivedTask(final Task task, final String userToken, final ItemChangeListener<Task> listener) {
        Error error = null;
        if (!isNetworkConnected()) {
            reportNetworkError(Action.UPDATE_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/data/task/" + task.getTid();

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "update task response: " + response);
                if (response == null) {
                    reportNetworkError(Action.UPDATE_NET, R.string.network_update_is_archived_task_error, listener);
                    return;
                }

                Task updatedTask = null;
                try {
                    updatedTask = gson.fromJson(response, new TypeToken<Task>() {
                    }.getType());
                } catch (Exception ex) {
                    reportNetworkError(Action.UPDATE_NET, R.string.network_json_error, listener);
                    return;
                }

                Log.d(TAG, "update task received: " + updatedTask.toString());
                listener.itemChanged(new ItemChangeResult<Task>(null, updatedTask, Action.UPDATE_NET, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.UPDATE_NET, R.string.network_update_is_archived_task_error, listener);
            }
        };

        StringRequest request = new StringRequest(Request.Method.PUT, url, responseListener, errorListener) {
            @Override
            public byte[] getBody() throws AuthFailureError {
                String jsonStr = "{\"isArchived\":" + (!task.isArchived()) + "}";
                try {
                    return jsonStr.getBytes("utf-8");
                } catch (Exception ex) {
                    ex.printStackTrace();
                    return null;
                }
            }

            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<>();
                headers.put("Content-Type", "application/json");
                headers.put("user-token", userToken);
                return headers;
            }
        };

        requestQueue.add(request);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    public void getAllTask(String ownerId, final String userToken, final ItemChangeListener<Task> listener) {
        Error error = null;
        if (!isNetworkConnected()) {
            reportNetworkError(Action.GET_ALL_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/data/task?where=ownerId%3D%27" + ownerId + "%27";

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "get all task response: " + response);
                if (response == null) {
                    reportNetworkError(Action.GET_ALL_NET, R.string.network_get_all_task_error, listener);
                    return;
                }

                List<Task> ownersTasks = null;
                try {
                    ownersTasks = gson.fromJson(response, new TypeToken<List<Task>>() {
                    }.getType());
                } catch (Exception ex) {
                    reportNetworkError(Action.GET_ALL_NET, R.string.network_json_error, listener);
                    return;
                }

                Log.d(TAG, "get all tasks received: " + response);
                listener.itemChanged(new ItemChangeResult<Task>(ownersTasks, null, Action.GET_ALL_NET, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.GET_ALL_NET, R.string.network_get_all_task_error, listener);
            }
        };

        StringRequest request = new StringRequest(Request.Method.GET, url, responseListener, errorListener) {
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                Map<String, String> headers = new HashMap<>();
                headers.put("Content-Type", "application/json");
                headers.put("user-token", userToken);
                return headers;
            }
        };

        requestQueue.add(request);
    }

    private void reportNetworkError(Action action, int errorStringId, ItemChangeListener listener) {
        Error error = new Error(context.getString(errorStringId));
        listener.itemChanged(new ItemChangeResult(null, null, action, error));
    }

    private void printVolleyErrorDetails(VolleyError error) {
        NetworkResponse errResponse = (error != null) ? error.networkResponse : null;
        int statusCode = 0;
        String data = "";
        if (errResponse != null) {
            statusCode = errResponse.statusCode;
            byte[] bytes = errResponse.data;
            try {
                data = (bytes != null) ? new String(bytes, "utf-8") : "";
            } catch (UnsupportedEncodingException ex) {
                ex.printStackTrace();
            }
        }

        Log.e(TAG, "Volley error with status code " + statusCode + " received with this message: " + data);
    }

}
