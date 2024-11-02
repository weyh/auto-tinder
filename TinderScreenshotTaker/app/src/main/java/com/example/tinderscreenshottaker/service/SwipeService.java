package com.example.tinderscreenshottaker.service;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.content.Intent;
import android.graphics.Path;
import android.view.accessibility.AccessibilityEvent;

import com.example.tinderscreenshottaker.MainActivity;
import com.example.tinderscreenshottaker.util.ELog;

public class SwipeService extends AccessibilityService {
    private static final String TAG = "SwipeService";

    public static final String ACTION_SWIPE_LEFT = "com.example.tinderscreenshottaker.action.SWIPE_LEFT";
    public static final String ACTION_SWIPE_RIGHT = "com.example.tinderscreenshottaker.action.SWIPE_RIGHT";

    private int screenWidth, screenHeight;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        screenWidth = getResources().getDisplayMetrics().widthPixels;
        screenHeight = getResources().getDisplayMetrics().heightPixels;

        if (intent != null) {
            final String action = intent.getAction();
            if (ACTION_SWIPE_LEFT.equals(action)) {
                performSwipeLeft();
            } else if (ACTION_SWIPE_RIGHT.equals(action)) {
                performSwipeRight();
            }
        }
        return START_NOT_STICKY;
    }

    private void performSwipeLeft() {
        ELog.d(TAG, "Performing swipe left");
        final Path path = new Path();
        // Swipe from near the right edge to the left
        path.moveTo(screenWidth - 100, screenHeight / 2);  // Start near the right edge
        path.lineTo(100, screenHeight / 2);                // End near the left edge
        performSwipe(path);
    }

    private void performSwipeRight() {
        ELog.d(TAG, "Performing swipe right");
        final Path path = new Path();
        // Swipe from near the left edge to the right
        path.moveTo(100, screenHeight / 2);                // Start near the left edge
        path.lineTo(screenWidth - 100, screenHeight / 2);  // End near the right edge
        performSwipe(path);
    }

    private void performSwipe(final Path dir) {
        ELog.d(TAG, "Performing swipe");
        final GestureDescription.Builder gestureBuilder = new GestureDescription.Builder();

        gestureBuilder.addStroke(new GestureDescription.StrokeDescription(dir, 100, 500));
        boolean success = dispatchGesture(gestureBuilder.build(), new GestureResultCallback() {
            @Override
            public void onCompleted(GestureDescription gestureDescription) {
                super.onCompleted(gestureDescription);
                ELog.d(TAG, "Swipe completed.");

                sendBroadcast(new Intent(MainActivity.MyBroadcastReceiver.INVOKE_AUTO_SWIPE_NEXT));
            }

            @Override
            public void onCancelled(GestureDescription gestureDescription) {
                super.onCancelled(gestureDescription);
                ELog.d(TAG, "Swipe cancelled.");
            }
        }, null);
        ELog.d(TAG, "Gesture dispatched: " + success);
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent accessibilityEvent) {

    }

    @Override
    public void onInterrupt() {

    }
}
