package com.example.tinderscreenshottaker;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.os.Vibrator;
import android.os.VibratorManager;
import android.provider.Settings;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.accessibility.AccessibilityManager;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;
import android.content.BroadcastReceiver;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.tinderscreenshottaker.service.FloatService;
import com.example.tinderscreenshottaker.service.PredictionFloatService;
import com.example.tinderscreenshottaker.service.ScreenRecordService;
import com.example.tinderscreenshottaker.service.ScreenshotImgType;
import com.example.tinderscreenshottaker.service.SwipeService;
import com.example.tinderscreenshottaker.util.AccessibilityUtil;
import com.example.tinderscreenshottaker.util.FileWriterUtil;
import com.example.tinderscreenshottaker.util.PredictClient;
import com.example.tinderscreenshottaker.util.VibrationEffectUtil;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import kotlin.Pair;

public class MainActivity extends AppCompatActivity {
    private static final int CODE_DRAW_OVER_OTHER_APP_PERMISSION = 1000;
    private static final int CODE_REQUEST_CAPTURE = 1001;

    private final String TAG = this.getClass().getSimpleName();

    private SharedPreferences sharedPreferences;

    private Vibrator vibrator;

    private MediaProjectionManager projectionManager;
    private boolean isRecording;
    private MyBroadcastReceiver myBroadcastReceiver;

    private ScreenshotImgType target = ScreenshotImgType.Unknown;

    private Map<ScreenshotImgType, ImageView> imgViews;

    private FileWriterUtil fileWriterUtil;

    private Handler backendHandler;
    private PredictClient predictClient;

    private boolean autoSwipeEnabled = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        myBroadcastReceiver = new MyBroadcastReceiver(this);
        imgViews = new HashMap<>();

        try {
            fileWriterUtil = new FileWriterUtil(this, "tinder_screenshot_taker");
        } catch (Exception e) {
            Toast.makeText(this, "Delete folder 'tinder_screenshot_taker'", Toast.LENGTH_LONG).show();
            Log.e(TAG, "Failed to create file writer", e);
            finish();
        }

        sharedPreferences = getSharedPreferences("settings", MODE_PRIVATE);

        HandlerThread handlerThread = new HandlerThread("TFThread", Process.THREAD_PRIORITY_LESS_FAVORABLE);
        handlerThread.start();
        backendHandler = new Handler(handlerThread.getLooper());

        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        final var vibratorManager = (VibratorManager) getSystemService(VIBRATOR_MANAGER_SERVICE);
        vibrator = vibratorManager.getDefaultVibrator();

        if (!AccessibilityUtil.isServiceEnabled(this, SwipeService.class)) {
            showAccessibilityPrompt();
        }

        if (!Settings.canDrawOverlays(this)) {
            // If the draw over permission is not available open the settings screen
            // to grant the permission.
            Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName()));
            startActivityForResult(intent, CODE_DRAW_OVER_OTHER_APP_PERMISSION);
        } else {
            start();
        }
    }

    private void showAccessibilityPrompt() {
        new MaterialAlertDialogBuilder(this)
                .setTitle("Accessibility Permission Required")
                .setMessage("To use the auto swipe feature, you need to enable Accessibility Service for this app. Please enable it in the next screen.")
                .setPositiveButton("Enable", (dialog, which) -> {
                    // Redirect to accessibility settings
                    Intent intent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
                    startActivity(intent);
                })
                .setNegativeButton("Cancel", (dialog, which) -> {
                    autoSwipeEnabled = false;
                    Toast.makeText(this, "Auto swipe cannot be enabled",
                            Toast.LENGTH_SHORT).show();
                })
                .show();
    }

    private void start() {
        projectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        registerReceiver(myBroadcastReceiver, myBroadcastReceiver.getFilter(), Context.RECEIVER_EXPORTED);

        initializeView();
    }

    private void initializeView() {
        final var ip = sharedPreferences.getString("ip", getString(R.string.DefaultIpDevInput));
        final var port = sharedPreferences.getString("port", getString(R.string.DefaultPort));
        autoSwipeEnabled = sharedPreferences.getBoolean("autoSwipe",
                Boolean.getBoolean(getString(R.string.DefaultAutoSwipeEnabled)));

        final EditText ipInput = findViewById(R.id.ipInput);
        ipInput.setText(ip);
        ipInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                final SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putString("ip", s.toString());
                editor.apply();
            }

            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) { }

            @Override
            public void afterTextChanged(Editable editable) {  }
        });

        final EditText portInput = findViewById(R.id.portInput);
        portInput.setText(port);
        portInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                final SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putString("port", s.toString());
                editor.apply();
            }

            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) { }

            @Override
            public void afterTextChanged(Editable editable) {  }
        });

        imgViews.put(ScreenshotImgType.Ok, (ImageView) findViewById(R.id.imageView));
        imgViews.put(ScreenshotImgType.X, (ImageView) findViewById(R.id.imageView2));

        findViewById(R.id.start_btn).setOnClickListener(view -> {
            if (!isRecording) {
                vibrator.vibrate(VibrationEffectUtil.MS_10);
                startActivityForResult(projectionManager.createScreenCaptureIntent(), CODE_REQUEST_CAPTURE);
            }
        });

        final CheckBox autoSwipeCheckBox = findViewById(R.id.autoSwipe);
        autoSwipeCheckBox.setChecked(autoSwipeEnabled);
        autoSwipeCheckBox.setOnClickListener(view -> {
            autoSwipeEnabled = ((CheckBox) view).isChecked();
            final SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putBoolean("autoSwipe", autoSwipeEnabled);
            editor.apply();
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch (requestCode) {
            case CODE_DRAW_OVER_OTHER_APP_PERMISSION:
                if (Settings.canDrawOverlays(this)) {
                    start();
                } else { //Permission is not available
                    Toast.makeText(this,
                            "Draw over other app permission not available. Closing the application",
                            Toast.LENGTH_SHORT).show();
                    Log.e(TAG, "Draw over other app permission not available. Closing the application");
                    finish();
                }
            case CODE_REQUEST_CAPTURE:
                if (resultCode == RESULT_OK) {
                    try {
                        final String ip = sharedPreferences.getString("ip", getString(R.string.DefaultIpDevInput));
                        final int port = Integer.parseInt(sharedPreferences.getString("port", getString(R.string.DefaultPort)));

                        predictClient = new PredictClient(ip, port);

                        startService(new Intent(MainActivity.this, FloatService.class));
                        startService(new Intent(MainActivity.this, PredictionFloatService.class));

                        Intent startIntent = new Intent(this, ScreenRecordService.class);
                        startIntent.setAction(ScreenRecordService.ACTION_START);
                        startIntent.putExtra(ScreenRecordService.EXTRA_RESULT_CODE, resultCode);
                        startIntent.putExtra(ScreenRecordService.EXTRA_DATA, data);
                        startService(startIntent);
                        isRecording = true;
                    } catch (RuntimeException e) {
                        Log.e(TAG, "Failed to get MediaProjection", e);
                        finish();
                    }
                } else {
                    Toast.makeText(this, "Screencast permission denied", Toast.LENGTH_SHORT).show();
                    Log.w(TAG, "Screencast permission denied. Closing the application");
                }
            default:
                break;
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    @SuppressLint("DefaultLocale")
    public void presentScreenshot(final byte[] bitmapData) {
        if (!isRecording)
            return;

        final Bitmap bitmap = BitmapFactory.decodeByteArray(bitmapData, 0, bitmapData.length);

        if (target == ScreenshotImgType.AI) {
            backendHandler.post(() -> {
                Pair<String, Float> ret;
                try {
                    ret = predictClient.predict(bitmapData);
                } catch (IllegalAccessError e) {
                    Log.w(TAG, "Failed to predict retry");
                    try {
                        ret = predictClient.predict(bitmapData);
                    } catch (IllegalAccessError e2) {
                        Log.e(TAG, "Failed to predict", e);
                        Toast.makeText(this, "Prediction failed (connerr)", Toast.LENGTH_LONG).show();
                        return;
                    }
                }
                Log.d(TAG, String.valueOf(ret));

                final Intent prediction = new Intent(this, PredictionFloatService.class);
                prediction.setAction(PredictionFloatService.ACTION_DISPLAY);
                prediction.putExtra(PredictionFloatService.EXTRA_TYPE, ret.getFirst().charAt(0));
                prediction.putExtra(PredictionFloatService.EXTRA_DATA, String.format("%s - %.2f%%", ret.getFirst(), ret.getSecond()));
                startService(prediction);

                if (autoSwipeEnabled) {
                    final AccessibilityManager accessibilityManager = (AccessibilityManager) getSystemService(ACCESSIBILITY_SERVICE);
                    if (!accessibilityManager.isTouchExplorationEnabled()) {
                        Log.e(TAG, "Touch exploration not enabled. Gesture injection may not work.");
                    }

                    final Intent swipeIntent = new Intent(this, SwipeService.class);
                    if (ret.getFirst().charAt(0) == 'x') {
                        swipeIntent.setAction(SwipeService.ACTION_SWIPE_LEFT);
                        startService(swipeIntent);
                    } else if (ret.getFirst().charAt(0) == 'o') {
                        swipeIntent.setAction(SwipeService.ACTION_SWIPE_RIGHT);
                        startService(swipeIntent);
                    } else {
                        Log.e(TAG, "Unknown prediction: " + ret.getFirst());
                    }
                }
            });
            return;
        }

        if (target == ScreenshotImgType.Ok) {
            fileWriterUtil.writeJpegFileToDownloads(String.format("ok_%d.jpg", System.currentTimeMillis()), bitmapData);
        } else if (target == ScreenshotImgType.X) {
            fileWriterUtil.writeJpegFileToDownloads(String.format("x_%d.jpg", System.currentTimeMillis()), bitmapData);
        }

        ImageView v = imgViews.get(target);
        if (v != null) {
            v.setImageBitmap(bitmap);
        } else {
            Log.e(TAG, "Unknown target: " + target);
        }
    }

    public void initiateTakeScreenshot(int target) {
        if (!isRecording)
            return;

        this.target = ScreenshotImgType.Companion.fromInt(target);

        final Intent snapshotIntent = new Intent(MainActivity.this, ScreenRecordService.class);
        snapshotIntent.setAction(ScreenRecordService.ACTION_SNAPSHOT);
        startService(snapshotIntent);
    }

    public void initiateStopRecording() {
        if (!isRecording)
            return;

        final Intent stopIntent = new Intent(this, ScreenRecordService.class);
        stopIntent.setAction(ScreenRecordService.ACTION_STOP);
        startService(stopIntent);

        final Intent closeIntent = new Intent(this, PredictionFloatService.class);
        closeIntent.setAction(PredictionFloatService.ACTION_STOP);
        startService(closeIntent);

        isRecording = false;
        target = ScreenshotImgType.Unknown;
    }

    @Override
    protected void onDestroy() {
        initiateStopRecording();
        unregisterReceiver(myBroadcastReceiver);

        super.onDestroy();
    }

    public static class MyBroadcastReceiver extends BroadcastReceiver {
        private final String TAG = this.getClass().getSimpleName();

        private static final String BASE = "com.example.tinderscreenshottaker.broadcast";
        public static final String INVOKE_SCREENSHOT = BASE + ".INVOKE_SCREENSHOT";
        public static final String INVOKE_STOP_RECORDING = BASE + ".INVOKE_STOP_RECORDING";
        public static final String INVOKE_SCREENSHOT_TAKEN = BASE + ".INVOKE_SCREENSHOT_TAKEN";
        public static final String INVOKE_AUTO_SWIPE_NEXT = BASE + ".INVOKE_AUTO_SWIPE_NEXT";

        private final IntentFilter filter;
        private final MainActivity ref;

        public MyBroadcastReceiver(final MainActivity ref) {
            super();
            this.ref = ref;
            this.filter = new IntentFilter();

            this.filter.addAction(INVOKE_SCREENSHOT);
            this.filter.addAction(INVOKE_STOP_RECORDING);
            this.filter.addAction(INVOKE_SCREENSHOT_TAKEN);
            this.filter.addAction(INVOKE_AUTO_SWIPE_NEXT);
        }

        public IntentFilter getFilter() {
            return filter;
        }

        @Override
        public void onReceive(Context context, Intent intent) {
            switch (Objects.requireNonNull(intent.getAction())) {
                case INVOKE_SCREENSHOT -> {
                    int data = intent.getIntExtra("data", -1);
                    ref.initiateTakeScreenshot(data);
                }
                case INVOKE_STOP_RECORDING -> {
                    Toast.makeText(context, "Recording has been stopped", Toast.LENGTH_SHORT).show();

                    ref.initiateStopRecording();
                }
                case INVOKE_SCREENSHOT_TAKEN -> {
                    final byte[] bitmapData = intent.getByteArrayExtra(ScreenRecordService.EXTRA_BITMAP);

                    if (bitmapData != null) {
                        ref.presentScreenshot(bitmapData);
                    } else {
                        Log.e(TAG, "Bitmap data is null (how did you get here)");
                        throw new NullPointerException();
                    }
                }
                case INVOKE_AUTO_SWIPE_NEXT -> {
                    if (!ref.autoSwipeEnabled) {
                        Log.d(TAG, "Auto swipe is disabled");
                        return;
                    } else if (!ref.isRecording) {
                        Log.d(TAG, "Not recording");
                        return;
                    }

                    // loops back to INVOKE_SCREENSHOT
                    final Intent broadcastIntent = new Intent(MainActivity.MyBroadcastReceiver.INVOKE_SCREENSHOT);
                    broadcastIntent.putExtra("data", ScreenshotImgType.Companion.toInt(ScreenshotImgType.AI));
                    ref.sendBroadcast(broadcastIntent);
                }
                default ->
                        Log.e(TAG, "Unknown action: " + intent.getAction());
            }
        }
    }
}
