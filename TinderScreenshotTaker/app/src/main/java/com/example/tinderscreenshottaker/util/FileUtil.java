package com.example.tinderscreenshottaker.util;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import java.io.IOException;
import java.io.OutputStream;

public class FileUtil {
    private static final String TAG = FileUtil.class.getSimpleName();
    private static Context ctx;

    public static void init(final Context context) {
        if (ctx != null) {
            ELog.w(TAG, "FileUtil already initialized");
        }
        ctx = context;
    }

    public static void createDirInDownloads(final WorkingDir env) {
        final var values = new ContentValues();
        values.put(MediaStore.Downloads.DISPLAY_NAME, env.dirName);
        values.put(MediaStore.Downloads.MIME_TYPE, "vnd.android.document/directory");
        values.put(MediaStore.Downloads.RELATIVE_PATH, env.workingDir);

        // Check if the directory already exists
        final var uri = ctx.getContentResolver()
                            .insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);

        if (uri != null) {
            ELog.d(TAG, "Directory created or already exists" + uri);
        } else {
            ELog.w(TAG, "Directory already exists or failed to create.");
        }
    }

    public static void writeJpegFile(final WorkingDir env, final String fileName, final byte[] content) throws IOException {
        writeFile(env, fileName, content, "image/jpeg");
    }

    public static void writeFile(final WorkingDir env, final String fileName, final byte[] content, final String mimeType) throws IOException {
        final var resolver = ctx.getContentResolver();
        final var values = new ContentValues();
        values.put(MediaStore.Downloads.DISPLAY_NAME, fileName);
        values.put(MediaStore.Downloads.MIME_TYPE, mimeType);
        values.put(MediaStore.Downloads.RELATIVE_PATH, env.workingDir);

        final var uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
        if (uri != null) {
            ELog.d(TAG, "Writing file to " + uri);
            try (final var os = resolver.openOutputStream(uri)) {
                if (os == null) {
                    ELog.d(TAG, "File opened successfully");
                    throw new IOException("File opened successfully");
                }

                os.write(content);
            }
        }
    }

    public static OutputStream createFileStream(final WorkingDir env, final String fileName, final String mimeType, final String mode) throws IOException {
        ContentResolver resolver = ctx.getContentResolver();
        ContentValues values = new ContentValues();

        values.put(MediaStore.Downloads.DISPLAY_NAME, fileName);
        values.put(MediaStore.Downloads.MIME_TYPE, mimeType);
        values.put(MediaStore.Downloads.RELATIVE_PATH, env.workingDir);

        final Uri uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);

        if (uri != null) {
            return resolver.openOutputStream(uri, mode);
        }

        throw new IOException("Failed to create file");
    }

    public static class WorkingDir {
        public final String dirName;
        public final String workingDir;

        public WorkingDir(String dirName) {
            this.dirName = dirName;
            this.workingDir = Environment.DIRECTORY_DOWNLOADS + "/" + dirName;
        }
    }
}
