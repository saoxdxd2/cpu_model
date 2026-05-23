#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <unistd.h>
#endif

/**
 * Aether AI IDE Bootstrapper
 * Orchestrates the full Silicon-to-Human stack in one click.
 */

void launch_engine() {
    std::cout << "[BOOT] Starting Aether Silicon Engine Gateway...\n";
#ifdef _WIN32
    // Gateway is in the SAME directory as the bootstrapper
    system("start /B aether_gateway.exe");
#else
    system("./aether_gateway &");
#endif
}

void open_dashboard() {
    std::cout << "[BOOT] Opening Aether Dashboard GUI...\n";
    // Point to the hosted React dashboard (Local or Remote)
    std::string url = "http://localhost:3000"; 
#ifdef _WIN32
    ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
#else
    std::string cmd = "open " + url;
    system(cmd.c_str());
#endif
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " NCA AETHER AI IDE — FULL STACK BOOTSTRAPPER          \n";
    std::cout << "========================================================\n\n";

    try {
        // 1. Launch Unified Silicon Engine (Handles Port 3001)
        launch_engine();
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // 2. Open User Interface (Points directly to Silicon)
        open_dashboard();

        std::cout << "\n[SUCCESS] Aether Stack is now fully operational.\n";
        std::cout << "[INFO] Keep this window open to maintain background services.\n";
        std::cout << "[INFO] Press Enter to shutdown the AI IDE stack.\n";

        std::cin.get();

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Boot Sequence collapsed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[BOOT] Shutting down Aether stack...\n";
#ifdef _WIN32
    system("taskkill /F /IM aether_gateway.exe /T");
#endif

    return 0;
}
