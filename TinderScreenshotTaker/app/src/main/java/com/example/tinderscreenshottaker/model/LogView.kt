package com.example.tinderscreenshottaker.model

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class LogView : ViewModel() {
    private val _text = MutableLiveData("")
    val text: LiveData<String> get() = _text

    @Synchronized
    fun setText(newText: String) {
        _text.postValue(newText)
    }

    @Synchronized
    fun appendText(newText: String) {
        val updatedText = (_text.value ?: "") + newText;
        _text.postValue(updatedText)
    }

    @Synchronized
    fun clearText() {
        setText("");
    }
}
