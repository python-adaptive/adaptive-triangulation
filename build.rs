use std::process::Command;

fn main() {
    let python = std::env::var("PYO3_PYTHON")
        .or_else(|_| std::env::var("PYTHON_SYS_EXECUTABLE"))
        .unwrap_or_else(|_| "python3".to_owned());

    let Ok(output) = Command::new(&python)
        .args(["-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')"])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }

    let Ok(libdir) = String::from_utf8(output.stdout) else {
        return;
    };
    let libdir = libdir.trim();
    if !libdir.is_empty() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }
}
