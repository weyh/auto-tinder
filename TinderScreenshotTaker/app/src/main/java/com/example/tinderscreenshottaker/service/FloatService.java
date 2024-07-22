package com.example.tinderscreenshottaker.service;

import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.graphics.drawable.Drawable;
import android.os.IBinder;
import android.os.VibratorManager;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;

import androidx.core.content.res.ResourcesCompat;

import com.example.tinderscreenshottaker.MainActivity;
import com.example.tinderscreenshottaker.R;
import com.example.tinderscreenshottaker.util.VibrationEffectUtil;

public class FloatService extends Service {
    private WindowManager windowManager;
    private View floatView;

    @Override
    public void onCreate() {
        super.onCreate();

        final var vibratorManager = (VibratorManager) getSystemService(VIBRATOR_MANAGER_SERVICE);
        final var vibrator = vibratorManager.getDefaultVibrator();

        // Inflate the float layout we created
        floatView = LayoutInflater.from(this).inflate(R.layout.layout_float, null);

        // Add the view to the window.
        final WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT);
        // Specify the chat head position
        params.gravity = Gravity.CENTER | Gravity.START;
        params.x = 0;
        params.y = 100 * 4;

        // Add the view to the window
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        windowManager.addView(floatView, params);

        //Set the close button.
        final Drawable closeNormalDrawable = ResourcesCompat.getDrawable(getResources(), R.drawable.ic_close, null);
        final Drawable closePressedDrawable = ResourcesCompat.getDrawable(getResources(), R.drawable.ic_close_p, null);
        final ImageView closeButton = floatView.findViewById(R.id.close_btn);

        createBtnActions(closeButton, v -> {
            vibrator.vibrate(VibrationEffectUtil.MS_10);

            final Intent broadcastIntent = new Intent(MainActivity.MyBroadcastReceiver.INVOKE_STOP_RECORDING);
            sendBroadcast(broadcastIntent);
            stopSelf();
        }, closeNormalDrawable, closePressedDrawable);

        final Drawable aiNormalDrawable = ResourcesCompat.getDrawable(getResources(), R.drawable.ic_ai, null);
        final Drawable aiPressedDrawable = ResourcesCompat.getDrawable(getResources(), R.drawable.ic_ai_p, null);
        final ImageView aiBtnImage = floatView.findViewById(R.id.ai_btn);

        createBtnActions(aiBtnImage, v -> {
            vibrator.vibrate(VibrationEffectUtil.MS_10);
            invokeScreenshot(ScreenshotImgType.AI);
        }, aiNormalDrawable, aiPressedDrawable);

        final Drawable xNormalDrawable = ResourcesCompat.getDrawable(getResources(), R.drawable.ic_x, null);
        final Drawable xPressedDrawable = ResourcesCompat.getDrawable(getResources(), R.drawable.ic_x_p, null);
        final ImageView xBtnImage = floatView.findViewById(R.id.x_btn);

        createBtnActions(xBtnImage, v -> {
            vibrator.vibrate(VibrationEffectUtil.MS_10);
            invokeScreenshot(ScreenshotImgType.X);
        }, xNormalDrawable, xPressedDrawable);

        final Drawable okNormalDrawable = ResourcesCompat.getDrawable(getResources(), R.drawable.ic_ok, null);
        final Drawable okPressedDrawable = ResourcesCompat.getDrawable(getResources(), R.drawable.ic_ok_p, null);
        final ImageView okBtnImage = floatView.findViewById(R.id.ok_btn);

        createBtnActions(okBtnImage, v -> {
            vibrator.vibrate(VibrationEffectUtil.MS_01);
            invokeScreenshot(ScreenshotImgType.Ok);
        }, okNormalDrawable, okPressedDrawable);
    }

    private void invokeScreenshot(final ScreenshotImgType data) {
        final Intent broadcastIntent = new Intent(MainActivity.MyBroadcastReceiver.INVOKE_SCREENSHOT);
        broadcastIntent.putExtra("data", ScreenshotImgType.Companion.toInt(data));
        sendBroadcast(broadcastIntent);
    }

    @SuppressLint("ClickableViewAccessibility")
    private static void createBtnActions(final ImageView btn, View.OnClickListener cb,
                                         final Drawable normalDrawable, final Drawable pressedDrawable) {
        btn.setOnTouchListener((v, e) -> {
            switch (e.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    btn.setImageDrawable(pressedDrawable);
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    cb.onClick(v);
                    btn.setImageDrawable(normalDrawable);
                    break;
            }
            return true;
        });

        btn.setImageDrawable(normalDrawable);
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
