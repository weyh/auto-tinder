package com.example.tinderscreenshottaker.util

import android.content.ContentValues
import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast

class FileWriterUtil(private val context: Context, private val dir: String) {
    init {
        createDir()
    }

    private fun createDir() {
        val values = ContentValues()
        values.put(MediaStore.Downloads.DISPLAY_NAME, dir)
        values.put(MediaStore.Downloads.MIME_TYPE, "vnd.android.document/directory")
        values.put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/" + dir)

        // Check if the directory already exists
        val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)

        if (uri != null) {
            Log.d("FileUtil", "Directory created or already exists: $uri")
        } else {
            Log.d("FileUtil", "Directory already exists or failed to create.")
        }
    }

    fun writeJpegFileToDownloads(fileName: String, content: ByteArray) {
        writeFileToDownloads(fileName, content, "image/jpeg")
    }

    fun writeFileToDownloads(fileName: String, content: ByteArray, mimeType: String) {
        val resolver = context.contentResolver
        val values = ContentValues()
        values.put(MediaStore.Downloads.DISPLAY_NAME, fileName)
        values.put(MediaStore.Downloads.MIME_TYPE, mimeType)
        values.put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/" + dir)

        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
        if (uri != null) {
            try {
                resolver.openOutputStream(uri).use { os ->
                    os?.write(content)
                }
            } catch (e: Exception) {
                Log.e("FileUtil", "Failed to write file", e)
                Toast.makeText(context, "Failed to write file", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
