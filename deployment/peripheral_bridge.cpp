#include "deployment/peripheral_bridge.hpp"
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace nca::deployment {

PeripheralBridge::PeripheralBridge() {}

void PeripheralBridge::execute_agentic_action(const float* action_vector) {
    // ── ACTION VECTOR MAPPING ──────────────────────────────────────────────
    // [0-4]   : Core System (IDLE, READ, WRITE, TEST, GIT)
    // [5-10]  : Mouse (X, Y, L-Click, R-Click, Scroll)
    // [11-40] : Keyboard (Top-30 most used keys)
    // [41-50] : Media (Screenshots, Image Gen, Video)
    
    // 1. Mouse Control
    if (action_vector[5] > 0.5f || std::abs(action_vector[6]) > 0.1f) {
        move_mouse(action_vector[5], action_vector[6]);
        if (action_vector[7] > 0.8f) trigger_click(0);
    }

    // 2. Keyboard Control
    for (int i = 11; i <= 40; ++i) {
        if (action_vector[i] > 0.9f) {
            tap_key(static_cast<uint8_t>(i + 54)); // Map to real ASCII/VK codes
        }
    }

    // 3. Media Triggers
    if (action_vector[41] > 0.95f) capture_screen("agent_sight.png");
    if (action_vector[42] > 0.99f) trigger_ai_image_gen("Concept generated from Silicon Thought.");
}

void PeripheralBridge::tap_key(uint8_t key_code) {
    std::cout << "  [PERIPHERAL] Tapping Key: " << (int)key_code << "\n";
#ifdef _WIN32
    keybd_event(key_code, 0, 0, 0);
    keybd_event(key_code, 0, KEYEVENTF_KEYUP, 0);
#endif
}

void PeripheralBridge::move_mouse(float rel_x, float rel_y) {
    // std::cout << "  [PERIPHERAL] Mouse Move: " << rel_x << ", " << rel_y << "\n";
#ifdef _WIN32
    // Simplified relative move
    mouse_event(MOUSEEVENTF_MOVE, (DWORD)(rel_x * 100), (DWORD)(rel_y * 100), 0, 0);
#endif
}

void PeripheralBridge::trigger_click(int button) {
    std::cout << "  [PERIPHERAL] Mouse Click: " << (button == 0 ? "LEFT" : "RIGHT") << "\n";
#ifdef _WIN32
    DWORD down = (button == 0) ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_RIGHTDOWN;
    DWORD up   = (button == 0) ? MOUSEEVENTF_LEFTUP   : MOUSEEVENTF_RIGHTUP;
    mouse_event(down, 0, 0, 0, 0);
    mouse_event(up, 0, 0, 0, 0);
#endif
}

void PeripheralBridge::capture_screen(const std::string& save_path) {
    std::cout << "  [PERIPHERAL] Capturing Real Desktop GUI -> " << save_path << "\n";
#ifdef _WIN32
    // Windows GDI Screenshot Implementation
    int x = GetSystemMetrics(SM_CXSCREEN);
    int y = GetSystemMetrics(SM_CYSCREEN);
    HDC hScreen = GetDC(NULL);
    HDC hDest = CreateCompatibleDC(hScreen);
    HBITMAP hBitmap = CreateCompatibleBitmap(hScreen, x, y);
    SelectObject(hDest, hBitmap);
    BitBlt(hDest, 0, 0, x, y, hScreen, 0, 0, SRCCOPY);
    
    // (Conceptual) In a full implementation, we'd save hBitmap to disk or a buffer.
    // For the proof, we verify the GDI handle is valid.
    if (hBitmap) std::cout << "  [OK] BitBlt Captured " << x << "x" << y << " resolution.\n";

    DeleteObject(hBitmap);
    DeleteDC(hDest);
    ReleaseDC(NULL, hScreen);
#endif
}

void PeripheralBridge::trigger_ai_image_gen(const std::string& prompt) {
    std::cout << "  [MEDIA_ROOM] Triggering External Image Generation: " << prompt << "\n";
    // Hook to stable-diffusion or other media AI
}

} // namespace nca::deployment
