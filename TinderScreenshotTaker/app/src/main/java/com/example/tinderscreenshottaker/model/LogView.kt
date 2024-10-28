package com.example.tinderscreenshottaker.model

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class LogView : ViewModel() {
    private val lock = Any()

    private val _text = MutableLiveData("")
    val text: LiveData<String> get() = _text

    var stringFormatter: ((String) -> String)? = null

    fun setFormater(f: (String) -> String) {
        synchronized(lock) {
            stringFormatter  = f
        }
    }

    private fun format(text: String): String {
        return stringFormatter?.invoke(text) ?: text
    }

    fun setText(newText: String) {
        synchronized(lock) {
            _text.postValue(format(newText))
        }
    }

    @Synchronized
    fun preappendText(newText: String) {
        synchronized(lock) {
            val updatedText = format(newText + (_text.value ?: ""));
            _text.postValue(updatedText)
        }
    }

    @Synchronized
    fun appendText(newText: String) {
        synchronized(lock) {
            val updatedText = format((_text.value ?: "") + newText);
            _text.postValue(updatedText)
        }
    }

    @Synchronized
    fun clearText() {
        setText("");
    }
}
