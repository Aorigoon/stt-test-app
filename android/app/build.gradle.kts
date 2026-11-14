plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.aorigoon.stt_test_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.aorigoon.stt_test_app"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        // onnxruntime_plus needs minSdk 24 (16 KB page-size .so, API 24+).
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        // Uses the version code from pubspec.yaml. When using split APKs, 1000 * ABI_VERSION
        // is added automatically by Flutter. (https://developer.android.com/studio/build/configure-apk-splits#configure-APK-versions)
        // You can force using the value of versionCode by specifying the `-P force-version-code-ignoring-abi=true`
        // flag during build.
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    // Release signing: keystore + passwords come from environment variables
    // (set by CI from GitHub secrets) or Gradle properties of the same names.
    // STT_TEST_KEYSTORE_FILE / STT_TEST_STORE_PASSWORD /
    // STT_TEST_KEY_PASSWORD / STT_TEST_KEY_ALIAS. No secrets in this file.
    signingConfigs {
        create("release") {
            val keystoreFile = System.getenv("STT_TEST_KEYSTORE_FILE")
                ?: project.findProperty("STT_TEST_KEYSTORE_FILE") as String?
            if (keystoreFile != null) {
                storeFile = file(keystoreFile)
            }
            storePassword = System.getenv("STT_TEST_STORE_PASSWORD")
                ?: project.findProperty("STT_TEST_STORE_PASSWORD") as String?
            keyAlias = System.getenv("STT_TEST_KEY_ALIAS")
                ?: project.findProperty("STT_TEST_KEY_ALIAS") as String?
            keyPassword = System.getenv("STT_TEST_KEY_PASSWORD")
                ?: project.findProperty("STT_TEST_KEY_PASSWORD") as String?
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
