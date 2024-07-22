package com.example.tinderscreenshottaker.service;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.Process;
import android.util.Log;

import androidx.core.app.NotificationCompat;

import com.example.tinderscreenshottaker.MainActivity;
import com.example.tinderscreenshottaker.R;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class ScreenRecordService extends Service {
    private static final String TAG = "ScreenRecordService";

    public static final String ACTION_START = "com.example.tinderscreenshottaker.action.START";
    public static final String ACTION_STOP = "com.example.tinderscreenshottaker.action.STOP";
    public static final String ACTION_SNAPSHOT = "com.example.tinderscreenshottaker.action.SNAPSHOT";

    public static final String EXTRA_RESULT_CODE = "RESULT_CODE";
    public static final String EXTRA_DATA = "DATA";
    public static final String EXTRA_BITMAP = "BITMAP";

    private final int NOTIFICATION_ID = 42;
    private final String CHANNEL_ID = "ScreenRecordChannel";

    private NotificationManager notificationManager;

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;
    private Handler backgroundHandler;

    private int density, width, height;

    @Override
    public void onCreate() {
        super.onCreate();
        notificationManager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        final NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Screen Recording Service Channel",
                NotificationManager.IMPORTANCE_DEFAULT
        );
        notificationManager.createNotificationChannel(channel);

        HandlerThread handlerThread = new HandlerThread("ScreenCapture", Process.THREAD_PRIORITY_VIDEO);
        handlerThread.start();
        backgroundHandler = new Handler(handlerThread.getLooper());

        final var metrics = getResources().getDisplayMetrics();
        height = 720; // metrics.heightPixels / 3;
        width = (int) (metrics.widthPixels / (float) (metrics.heightPixels / height));
        density = (int) (metrics.densityDpi / (float) (metrics.heightPixels / height));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null) {
            final String action = intent.getAction();

            if (ACTION_START.equals(action)) {
                final int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_OK);
                final Intent data = intent.getParcelableExtra(EXTRA_DATA);

                // Add a notification for the service
                startForeground(NOTIFICATION_ID, createNotification());

                startRecording(resultCode, data);
            } else if (ACTION_STOP.equals(action)) {
                stopRecording();
                stopForeground(true);
                stopSelf();
            } else if (ACTION_SNAPSHOT.equals(action)) {
                takeSnapshot();
            }
        }
        return START_NOT_STICKY;
    }

    private void startRecording(final int resultCode, final Intent data) {
        MediaProjectionManager projectionManager = (MediaProjectionManager) getSystemService(MEDIA_PROJECTION_SERVICE);
        mediaProjection = projectionManager.getMediaProjection(resultCode, data);
        mediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                super.onStop();
            }
        }, null);


        imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2);
        virtualDisplay = mediaProjection.createVirtualDisplay("ScreenCapture",
                width, height, density,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(), null, backgroundHandler);

    }

    private void stopRecording() {
        if (mediaProjection != null) {
            mediaProjection.stop();
            mediaProjection = null;
        }
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
    }

    private void takeSnapshot() {
        if (imageReader != null) {
            Image image = imageReader.acquireLatestImage();
            if (image != null) {
                Image.Plane[] planes = image.getPlanes();
                ByteBuffer buffer = planes[0].getBuffer();
                int pixelStride = planes[0].getPixelStride();
                int rowStride = planes[0].getRowStride();
                int rowPadding = rowStride - pixelStride * width;

                Bitmap bitmap = Bitmap.createBitmap(width + rowPadding / pixelStride, height, Bitmap.Config.ARGB_8888);
                bitmap.copyPixelsFromBuffer(buffer);

                image.close();

                final var byteArray = BitmapToJPEGByteArray(bitmap);

                final Intent broadcastIntent = new Intent(MainActivity.MyBroadcastReceiver.INVOKE_SCREENSHOT_TAKEN);
                broadcastIntent.putExtra(EXTRA_BITMAP, byteArray);
                sendBroadcast(broadcastIntent);
            }
        }
    }

    private Notification createNotification() {
        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Screen Recording")
                .setContentText("Recording your screen")
                .setSmallIcon(R.drawable.ic_ok)
                .build();
    }

    private static byte[] BitmapToJPEGByteArray(final Bitmap bitmap) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream);
        return stream.toByteArray();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        stopRecording();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
