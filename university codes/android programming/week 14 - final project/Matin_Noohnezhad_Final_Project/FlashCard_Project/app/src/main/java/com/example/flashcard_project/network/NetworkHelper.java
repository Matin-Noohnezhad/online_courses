package com.example.flashcard_project.network;

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
import com.example.flashcard_project.R;
import com.example.flashcard_project.beans.FlashCard;
import com.example.flashcard_project.beans.User;
import com.example.flashcard_project.utils.Action;
import com.example.flashcard_project.utils.ItemChangeListener;
import com.example.flashcard_project.utils.ItemChangeResult;
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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public void userSignUp(final String email, final String password, final String name, final ItemChangeListener<User> listener) {
        if (!isNetworkConnected()) {
            reportNetworkError(Action.SIGN_UP_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/users/register";

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
        StringRequest request = new StringRequest(Request.Method.POST, url, responseListener, errorListener) {
            @Override
            public byte[] getBody() throws AuthFailureError {
                try {
                    String jsonStr = "{\n" +
                            "\"email\":\"" + email + "\",\n" +
                            "\"password\":\"" + password + "\",\n" +
                            "\"name\":\"" + name + "\"\n" +
                            "}\n";
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

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public void checkUserTokenValidity(String userToken, final ItemChangeListener<Boolean> listener) {
        if (!isNetworkConnected()) {
            reportNetworkError(Action.CHECK_USER_TOKEN_VALIDITY_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/users/isvalidusertoken/" + userToken;

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "user-token validity response: " + response);
                if (response == null) {
                    reportNetworkError(Action.CHECK_USER_TOKEN_VALIDITY_NET, R.string.network_check_user_token_validity_error, listener);
                    return;
                }

                Log.d(TAG, "user-token validity received: " + response);
                listener.itemChanged(new ItemChangeResult<Boolean>(null, Boolean.parseBoolean(response), Action.CHECK_USER_TOKEN_VALIDITY_NET, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.CHECK_USER_TOKEN_VALIDITY_NET, R.string.network_check_user_token_validity_error, listener);
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

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public void addNewCard(FlashCard card, final String userToken, final ItemChangeListener<FlashCard> listener) {
        if (!isNetworkConnected()) {
            reportNetworkError(Action.INSERT_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/data/flashcards";

        String cardStr = null;
        try {
            cardStr = gson.toJson(card);
        } catch (Exception ex) {
            reportNetworkError(Action.INSERT_NET, R.string.network_json_error, listener);
            return;
        }

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "insert flashcard response: " + response);
                if (response == null) {
                    reportNetworkError(Action.INSERT_NET, R.string.network_add_new_task_error, listener);
                    return;
                }

                FlashCard insertedCard = null;
                try {
                    String utf8String = new String(response.getBytes("ISO-8859-1"), "UTF-8");
                    insertedCard = gson.fromJson(utf8String, new TypeToken<FlashCard>() {
                    }.getType());
                } catch (Exception ex) {
                    reportNetworkError(Action.INSERT_NET, R.string.network_json_error, listener);
                    return;
                }

                Log.d(TAG, "insert new flashcard received: " + insertedCard.toString());
                listener.itemChanged(new ItemChangeResult<FlashCard>(null, insertedCard, Action.INSERT_NET, null));
            }
        };

        Response.ErrorListener errorListener = new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError err) {
                printVolleyErrorDetails(err);
                reportNetworkError(Action.INSERT_NET, R.string.network_add_new_task_error, listener);
            }
        };

        final String jsonStr = cardStr;
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
                headers.put("user-token", userToken);
                return headers;
            }
        };

        requestQueue.add(request);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public void getAllTask(final String userToken, final int offset, final ItemChangeListener<FlashCard> listener) {
        if (!isNetworkConnected()) {
            reportNetworkError(Action.GET_ALL_NET, R.string.network_connection_error, listener);
            return;
        }

        String url = hostUrl + "/" + appId + "/" + apiKey + "/data/flashcards?pageSize=100&offset=" + String.valueOf(offset);

        Response.Listener<String> responseListener = new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Log.d(TAG, "get all flashcards response: " + response);
                if (response == null) {
                    reportNetworkError(Action.GET_ALL_NET, R.string.network_get_all_task_error, listener);
                    return;
                }

                List<FlashCard> allCards = null;
                try {
                    String utf8String = new String(response.getBytes("ISO-8859-1"), "UTF-8");
                    allCards = gson.fromJson(utf8String, new TypeToken<List<FlashCard>>() {
                    }.getType());
                } catch (Exception ex) {
                    reportNetworkError(Action.GET_ALL_NET, R.string.network_json_error, listener);
                    return;
                }

                Log.d(TAG, "get all flashcards received: " + response);
                listener.itemChanged(new ItemChangeResult<FlashCard>(allCards, null, Action.GET_ALL_NET, null));
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
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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
