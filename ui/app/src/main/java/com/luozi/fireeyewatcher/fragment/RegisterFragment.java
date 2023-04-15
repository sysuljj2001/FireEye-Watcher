package com.luozi.fireeyewatcher.fragment;

import android.content.Context;
import android.os.Bundle;

import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.fragment.app.FragmentTransaction;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.luozi.fireeyewatcher.R;
import com.luozi.fireeyewatcher.http.Common;
import com.luozi.fireeyewatcher.model.User;
import com.luozi.fireeyewatcher.utils.ToastCustom;

import org.apache.hc.client5.http.async.methods.SimpleHttpRequest;
import org.apache.hc.client5.http.async.methods.SimpleHttpResponse;
import org.apache.hc.client5.http.config.ConnectionConfig;
import org.apache.hc.client5.http.impl.async.CloseableHttpAsyncClient;
import org.apache.hc.client5.http.impl.async.HttpAsyncClients;
import org.apache.hc.client5.http.impl.nio.PoolingAsyncClientConnectionManager;
import org.apache.hc.client5.http.impl.nio.PoolingAsyncClientConnectionManagerBuilder;
import org.apache.hc.core5.concurrent.FutureCallback;
import org.apache.hc.core5.http.ContentType;
import org.apache.hc.core5.http.Method;
import org.json.JSONException;
import org.json.JSONObject;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.concurrent.TimeUnit;

public class RegisterFragment extends Fragment {

    private Context context;
    private View view;
    private CloseableHttpAsyncClient client;
    private EditText et_username;
    private EditText et_pswd;
    private EditText et_check_pswd;
    private Button btn_register;
    private TextView tv_back_to_login;
    private ProgressBar pb_register;
    private static final String LOG_TAG = "REGISTER_FRAGMENT";

    private class RegisterRequestCallback implements FutureCallback<SimpleHttpResponse> {

        @Override
        public void completed(SimpleHttpResponse httpResponse) {
            try {
                if (httpResponse == null) {
                    throw new RuntimeException("empty http response");
                }

                JSONObject jsonResponse = new JSONObject(httpResponse.getBodyText());

                int statusCode = jsonResponse.getInt("status_code");
                String desc = jsonResponse.getString("desc");
                String data = jsonResponse.getString("data");

                if (statusCode >= Common.STATUS_REQUEST_ERROR) {
                    ToastCustom.custom(context, "远程服务器异常");
                    throw new RuntimeException("remote server error");
                } else if (statusCode >= Common.STATUS_REQUEST_ERROR) {
                    ToastCustom.custom(context, String.format("请求错误: %S", desc));
                    throw new RuntimeException("register request error");
                }

                User user = User.parseFromJson(new JSONObject(data));
                Log.d(LOG_TAG, String.format("register successfully, username: %s", user.username));
                jumpToLogin();
            } catch (RuntimeException | JSONException e) {
                e.printStackTrace();
            } finally {
                pb_register.setVisibility(View.INVISIBLE);
                btn_register.setEnabled(true);
            }
        }

        @Override
        public void failed(Exception e) {
            Log.e(LOG_TAG, String.format("register request failed, error: %s", e.getLocalizedMessage()));
            ToastCustom.custom(context, "连接超时");
            pb_register.setVisibility(View.INVISIBLE);
            btn_register.setEnabled(true);
        }

        @Override
        public void cancelled() {
            Log.d(LOG_TAG, "register request has been cancelled");
            pb_register.setVisibility(View.INVISIBLE);
            btn_register.setEnabled(true);
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        context = getActivity();
        ConnectionConfig connectionConfig = ConnectionConfig.custom()
                .setConnectTimeout(5, TimeUnit.SECONDS)
                .setSocketTimeout(5, TimeUnit.SECONDS)
                .build();
        PoolingAsyncClientConnectionManager poolingAsyncClientConnectionManager =
                PoolingAsyncClientConnectionManagerBuilder.create()
                        .setDefaultConnectionConfig(connectionConfig)
                        .build();
        client = HttpAsyncClients.custom()
                .setConnectionManager(poolingAsyncClientConnectionManager)
                .build();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        view =  inflater.inflate(R.layout.fragment_register, container, false);
        et_username = view.findViewById(R.id.et_username);
        et_pswd = view.findViewById(R.id.et_pswd);
        et_check_pswd = view.findViewById(R.id.et_check_pswd);
        btn_register = view.findViewById(R.id.btn_register);
        tv_back_to_login = view.findViewById(R.id.tv_back_to_login);
        pb_register = view.findViewById(R.id.pb_register);

        btn_register.setOnClickListener(view -> {
            InputMethodManager imm = (InputMethodManager) context.getSystemService(Context.INPUT_METHOD_SERVICE);
            imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
            btn_register.setEnabled(false);

            String username = et_username.getEditableText().toString();
            String password = et_pswd.getEditableText().toString();
            String checkPassword = et_check_pswd.getEditableText().toString();

            if (username.isEmpty() || password.isEmpty()) {
                ToastCustom.custom(context, "账号或密码为空");
                return;
            }

            if (password.compareTo(checkPassword) != 0) {
                ToastCustom.custom(context, "两次输入密码不一致");
                return;
            }

            JSONObject data = new JSONObject();
            try {
                data.put("name", username);
                data.put("password", password);
            } catch (JSONException e) {
                throw new RuntimeException(e);
            }

            URI uri = null;
            try {
                uri = new URI("http://10.0.2.2:8080/api/v1/auth/register");
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            }
            SimpleHttpRequest request = new SimpleHttpRequest(Method.POST, uri);
            request.setHeader("Content-Type", "application/json");
            request.setHeader("Accept", "application/json");
            request.setBody(data.toString(), ContentType.APPLICATION_JSON);

            pb_register.setVisibility(View.VISIBLE);
            client.start();
            client.execute(request, new RegisterRequestCallback());
        });

        tv_back_to_login.setOnClickListener(view -> {
            jumpToLogin();
        });

        return view;
    }

    private void jumpToLogin() {
        FragmentTransaction transaction = ((FragmentActivity)context).getSupportFragmentManager().beginTransaction();
        LoginFragment loginFragment = new LoginFragment();
        transaction.replace(R.id.fragment_container_login, loginFragment);
        transaction.commitAllowingStateLoss();
    }
}