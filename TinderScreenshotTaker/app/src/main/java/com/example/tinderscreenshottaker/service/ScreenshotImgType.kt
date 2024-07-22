package com.example.tinderscreenshottaker.service

enum class ScreenshotImgType {
    Unknown, Ok, X, AI, Error;

    companion object {
        fun fromInt(i: Int): ScreenshotImgType {
            return when (i) {
                1 -> Ok
                2 -> X
                3 -> AI
                4 -> Error
                else -> Unknown
            }
        }

        fun toInt(t: ScreenshotImgType): Int {
            return when (t) {
                Unknown -> 0
                Ok -> 1
                X -> 2
                AI -> 3
                Error -> 4
            }
        }
    }
}
