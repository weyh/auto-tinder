package com.example.tinderscreenshottaker.util;

import android.annotation.SuppressLint;
import android.graphics.Color;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;
import android.util.Log;

import com.example.tinderscreenshottaker.model.LogView;

import java.text.SimpleDateFormat;
import java.util.Date;

public class ELog {
    @SuppressLint("SimpleDateFormat")
    private static final SimpleDateFormat sdf = new SimpleDateFormat("yy-MM-dd HH:mm:ss.SSS");

    private static LogView logView = null;

    private static final Object lock = new Object();

    public static void init(final LogView view) {
        synchronized (lock) {
            logView = view;
        }
    }

    public static void d(String tag, String msg) {
        log("DEBUG", tag, msg);
        Log.d(tag, msg);
    }

    public static void i(String tag, String msg) {
        log("INFO", tag, msg);
        Log.i(tag, msg);
    }

    public static void w(String tag, String msg) {
        log("WARN", tag, msg);
        Log.w(tag, msg);
    }

    public static void e(String tag, String msg) {
        log("ERROR", tag, msg);
        Log.e(tag, msg);
    }

    @SuppressLint("DefaultLocale")
    private static void log(String logType, final String tag, final String msg) {
        synchronized (lock) {
            if(logView != null) {
                logView.preappendText(String.format("%s %s [%s]:\t%s\n", logType, sdf.format(new Date()), tag, msg));
            }
        }
    }

    public static void close() {
        synchronized (lock) {
            logView = null;
        }
    }
}
