// uSTAT Desktop — Tauri v2 library entry point
//
// Architecture:
// 1. On launch, spawn the bundled Python backend (FastAPI/uvicorn) as a subprocess
// 2. Poll until the backend is ready via TCP connection
// 3. Navigate the webview to http://127.0.0.1:<port>
// 4. On close, kill the backend process

use tauri::Manager;

struct BackendProcess(std::sync::Mutex<Option<u32>>);

/// Find a free TCP port starting from `preferred`.
fn find_free_port(preferred: u16) -> u16 {
    for port in preferred..preferred + 50 {
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind to any port")
        .local_addr()
        .unwrap()
        .port()
}

fn kill_process(pid: u32) {
    #[cfg(unix)]
    {
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .output();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = find_free_port(18731);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .setup(move |app| {
            let handle = app.handle().clone();

            // Resolve the bundled backend binary path
            let resource_dir = handle
                .path()
                .resource_dir()
                .expect("cannot resolve resource dir");

            let backend_binary = if cfg!(target_os = "windows") {
                resource_dir.join("binaries").join("ustat-backend.exe")
            } else {
                resource_dir.join("binaries").join("ustat-backend")
            };

            // Spawn the backend process
            let port_str = port.to_string();
            let child = std::process::Command::new(&backend_binary)
                .args(["--port", &port_str])
                .env("USTAT_NO_BROWSER", "1")
                .env("USTAT_DESKTOP_MODE", "1")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap_or_else(|e| {
                    eprintln!("Failed to start uSTAT backend at {:?}: {}", backend_binary, e);
                    std::process::exit(1);
                });

            let child_id = child.id();
            app.manage(BackendProcess(std::sync::Mutex::new(Some(child_id))));

            // Spawn a thread to wait for the backend, then navigate
            std::thread::spawn(move || {
                let url = format!("http://127.0.0.1:{}", port);
                let max_wait = std::time::Duration::from_secs(30);
                let start = std::time::Instant::now();

                loop {
                    if start.elapsed() > max_wait {
                        eprintln!("uSTAT backend did not start within 30s");
                        break;
                    }
                    if std::net::TcpStream::connect_timeout(
                        &format!("127.0.0.1:{}", port).parse().unwrap(),
                        std::time::Duration::from_millis(200),
                    )
                    .is_ok()
                    {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        if let Some(window) = handle.get_webview_window("main") {
                            let _ = window.navigate(url.parse().unwrap());
                        }
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(300));
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.try_state::<BackendProcess>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(pid) = guard.take() {
                            kill_process(pid);
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running uSTAT desktop");
}
