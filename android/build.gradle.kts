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

// onnxruntime_plus 1.5.0 hardcodes compileSdkVersion 33 in its
// android/build.gradle, but its transitive androidx dependencies
// (fragment 1.7.1, core 1.13.1, lifecycle 2.7.0, ...) require
// compileSdk 34+. The plugin's AAR metadata check then fails the build.
// Force compileSdk 36 on every Android module after evaluation.
// Reflection is used deliberately: AGP classes are not on this root
// script's classpath, so a typed LibraryExtension reference would fail
// script compilation.
//
// ORDERING MATTERS: this block must stay ABOVE the
// evaluationDependsOn(":app") block below. That block forces :app to
// evaluate during root-script evaluation, and afterEvaluate cannot be
// registered on an already-evaluated project ("Cannot run
// Project.afterEvaluate(Action) when the project is already evaluated").
// Registering the hook first also guarantees it runs before AGP's own
// afterEvaluate callbacks, so variants are created with compileSdk 36.
fun Project.forceCompileSdk36() {
    val androidExt = extensions.findByName("android") ?: return
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

subprojects {
    try {
        afterEvaluate { forceCompileSdk36() }
    } catch (e: IllegalStateException) {
        // Already evaluated (should not happen with the ordering above):
        // apply directly instead of failing configuration.
        forceCompileSdk36()
    }
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
