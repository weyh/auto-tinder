package com.example.tinderscreenshottaker.util;

import android.accessibilityservice.AccessibilityService;
import android.content.ComponentName;
import android.content.Context;
import android.provider.Settings;
import android.text.TextUtils;

public class AccessibilityUtil {
    public static boolean isServiceEnabled(Context context, Class<? extends AccessibilityService> service) {
        final ComponentName expectedComponentName = new ComponentName(context, service);

        final String enabledServices = Settings.Secure.getString(
                context.getContentResolver(),
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES);

        final TextUtils.SimpleStringSplitter splitter = new TextUtils.SimpleStringSplitter(':');

        if (enabledServices != null && !enabledServices.isEmpty()) {
            splitter.setString(enabledServices);
            while (splitter.hasNext()) {
                final String componentName = splitter.next();
                if (componentName.equalsIgnoreCase(expectedComponentName.flattenToString())) {
                    return true;
                }
            }
        }

        return false;
    }
}
