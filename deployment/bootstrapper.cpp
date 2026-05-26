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

#include "deployment/silicon_ui.hpp"

using namespace nca::deployment;

void launch_engine() {
    std::cout << "[BOOT] Starting Aether Silicon Engine Gateway...\n";
#ifdef _WIN32
    // Gateway is in the SAME directory as the bootstrapper
    system("start /B aether_gateway.exe");
#else
    system("./aether_gateway &");
#endif
}

void start_native_gui() {
    std::cout << "[BOOT] Opening NCA Aether Native GUI (Saturated Silicon)...\n";
    std::cout << "       (External browser suppression ACTIVE)\n";
    
    // Instead of opening a browser, we initialize the native renderer
    SiliconUI ui("NCA Aether IDE - Professional Silicon Ground");
    
    // The main thread now hosts the UI render loop
    while(true) {
        ui.render();
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " NCA AETHER AI IDE — FULL STACK BOOTSTRAPPER          \n";
    std::cout << "========================================================\n\n";

    try {
        // 1. Launch Unified Silicon Engine (Handles Port 3001 and UI)
        launch_engine();
        std::cout << "[SUCCESS] Aether Silicon Host started in background.\n";
        std::cout << "[INFO] Control the engine via the Dashboard (localhost:3000).\n";
        
        // The bootstrapper now just stays alive to monitor the gateway
        while(true) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }

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
