fn main() {
    // Tell cargo to rerun this if the Python executable changes
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    // On macOS, we need to link against the Python framework
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-search=/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib");
        println!("cargo:rustc-link-search=/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib/python3.13/config-3.13-darwin");
        println!("cargo:rustc-link-lib=python3.13");
    }
} 