package com.example.tinderscreenshottaker.service;

import android.app.Service;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.IBinder;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;

import com.example.tinderscreenshottaker.R;

public class PredictionFloatService extends Service {
    public static final String ACTION_STOP = "com.example.tinderscreenshottaker.action.STOP";
    public static final String ACTION_DISPLAY = "com.example.tinderscreenshottaker.action.DISPLAY";

    public static final String EXTRA_TYPE = "TYPE";
    public static final String EXTRA_DATA = "DATA";

    private WindowManager windowManager;
    private View floatView;

    private TextView result;

    @Override
    public void onCreate() {
        super.onCreate();

        // Inflate the float layout we created
        floatView = LayoutInflater.from(this).inflate(R.layout.layout_prediction, null);

        // Add the view to the window.
        final WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT);
        // Specify the chat head position
        params.gravity = Gravity.TOP | Gravity.END;
        params.x = 0;
        params.y = 0;

        // Add the view to the window
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        windowManager.addView(floatView, params);

        result = floatView.findViewById(R.id.result);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null) {
            String action = intent.getAction();
            if (ACTION_STOP.equals(action)) {
                stopSelf();
            } else if (ACTION_DISPLAY.equals(action)) {
                final char type = intent.getCharExtra(EXTRA_TYPE, '\0');
                final String data = intent.getStringExtra(EXTRA_DATA);

                if (type == '\0') {
                    Log.e(this.getClass().getSimpleName(), "Unknown type: " + type);
                    return START_NOT_STICKY;
                }

                result.setText(data);

                if (type == 'x') {
                    result.setTextColor(getResources().getColor(R.color.x));
                } else if (type == 'o') {
                    result.setTextColor(getResources().getColor(R.color.ok));
                } else {
                    Log.e(this.getClass().getSimpleName(), "Unknown type: " + type);
                }
            }
        }
        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (floatView != null) windowManager.removeView(floatView);
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
