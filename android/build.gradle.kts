allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// onnxruntime_plus 1.5.0 hardcodes compileSdkVersion 33 in its
// android/build.gradle, but its transitive androidx dependencies
// (fragment 1.7.1, core 1.13.1, lifecycle 2.7.0, ...) require
// compileSdk 34+. The plugin's AAR metadata check then fails the build.
// Force compileSdk 36 on every Android module after evaluation.
// Reflection is used deliberately: AGP classes are not on this root
// script's classpath, so a typed LibraryExtension reference would fail
// script compilation.
subprojects {
    afterEvaluate {
        val androidExt = extensions.findByName("android") ?: return@afterEvaluate
        val setter =
            androidExt.javaClass.methods.firstOrNull {
                it.name == "setCompileSdk" && it.parameterCount == 1
            }
                ?: androidExt.javaClass.methods.firstOrNull {
                    it.name == "setCompileSdkVersion" && it.parameterCount == 1
                }
        if (setter != null) {
            setter.invoke(androidExt, 36)
            println("stt-test-app: forced compileSdk 36 on ${project.path}")
        } else {
            println("stt-test-app: WARNING could not find compileSdk setter on ${project.path}")
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
