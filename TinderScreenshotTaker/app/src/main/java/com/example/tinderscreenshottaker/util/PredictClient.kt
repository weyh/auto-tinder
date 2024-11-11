package com.example.tinderscreenshottaker.util

import android.util.Base64
import org.kotlincrypto.endians.LittleEndian
import org.kotlincrypto.endians.LittleEndian.Companion.toLittleEndian
import java.net.Socket
import java.security.MessageDigest
import javax.crypto.Cipher
import javax.crypto.spec.SecretKeySpec

class PredictClient(private val ip: String, private val port: Int) {
    private companion object {
        const val KEY = "4-KEY_for_this+s3rveR"
        const val CHECK_DATA_BASE = "<0_w_0>"
    }

    private fun generateCheckData(): String {
        return "${CHECK_DATA_BASE}_${port}+${(System.currentTimeMillis() / (1000 * 60 * 60)).toLong()}"
    }

    private fun deriveKey(key: String): ByteArray {
        val digest = MessageDigest.getInstance("SHA-256")
        return digest.digest(key.toByteArray(Charsets.UTF_8)).copyOf(16)  // AES-128 uses 16 bytes key
    }

    private fun encrypt(input: String, key: String): String {
        val cipher = Cipher.getInstance("AES/ECB/PKCS5Padding")
        val secretKey = SecretKeySpec(deriveKey(key), "AES")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey)
        val encrypted = cipher.doFinal(input.toByteArray(Charsets.UTF_8))
        return Base64.encodeToString(encrypted, Base64.DEFAULT)
    }

    fun predict(imgData: ByteArray): Pair<String, Float> {
        val socket = Socket(ip, port)
        val output = socket.getOutputStream()
        val input = socket.getInputStream()

        try {
            val encCheckData = encrypt(generateCheckData(), KEY)
            val encCheckDataByteArray = encCheckData.toByteArray()
            val encCheckDataByteArrayLen = encCheckDataByteArray.size.toLittleEndian().toByteArray()
            output.write(encCheckDataByteArrayLen)
            output.write(encCheckDataByteArray)

            var resp = input.readNBytes(1)
            var data = resp.toString(Charsets.UTF_8)

            if (data == "n") {
                throw IllegalAccessException("Failed to authenticate")
            }

            val imgDataLen = imgData.size.toLittleEndian().toByteArray()
            output.write(imgDataLen)
            output.write(imgData)

            resp = input.readNBytes(1)
            data = resp.toString(Charsets.UTF_8)
            val type: String = if (data == "o") { "ok" } else { "x" }

            resp = input.readNBytes(2)
            val score: Float = LittleEndian.bytesToInt(resp[0], resp[1], 0, 0) / 100f
            return type to score
        } finally {
            input.close()
            output.close()
            socket.close()
        }
    }
}
