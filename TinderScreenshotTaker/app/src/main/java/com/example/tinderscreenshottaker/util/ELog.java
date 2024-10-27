package com.example.tinderscreenshottaker.util;

import android.annotation.SuppressLint;
import android.util.Log;

import com.example.tinderscreenshottaker.model.LogView;

public class ELog {
    private static LogView logView = null;

    private static final Object lock = new Object();

    public static void init(final LogView view) {
        synchronized (lock) {
            logView = view;
        }
    }

    public static void d(String tag, String msg) {
        log("D", tag, msg);
        Log.d(tag, msg);
    }

    public static void i(String tag, String msg) {
        log("I", tag, msg);
        Log.i(tag, msg);
    }

    public static void w(String tag, String msg) {
        log("W", tag, msg);
        Log.w(tag, msg);
    }

    public static void e(String tag, String msg) {
        log("E", tag, msg);
        Log.e(tag, msg);
    }

    @SuppressLint("DefaultLocale")
    private static void log(String logType, final String tag, final String msg) {
        synchronized (lock) {
            if(logView != null) {
                final long now = System.currentTimeMillis();
                logView.appendText(String.format("%s %d [%s]:\t%s\n", logType, now, tag, msg));
            }
        }
    }

    public static void close() {
        synchronized (lock) {
            logView = null;
        }
    }
}
