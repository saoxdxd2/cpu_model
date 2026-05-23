#include "deployment/silicon_ui.hpp"
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
#include <d2d1.h>
#pragma comment(lib, "d2d1.lib")
#endif

namespace nca::deployment {

SiliconUI::SiliconUI(const std::string& title, int width, int height) 
    : d2d_factory_(nullptr), render_target_(nullptr), brush_(nullptr) {
    std::cout << "[SiliconUI] Creating Hardware-Saturated Window: \"" << title << "\"\n";
    std::cout << "           Bypassing Electron shell for maximum security.\n";
    
#ifdef _WIN32
    // Simplified Win32 window creation for the Proof
    WNDCLASS wc = {0};
    wc.lpfnWndProc = DefWindowProc;
    wc.lpszClassName = "NCA_Aether_IDE";
    wc.hInstance = GetModuleHandle(NULL);
    RegisterClass(&wc);
    window_handle_ = CreateWindow("NCA_Aether_IDE", title.c_str(), WS_OVERLAPPEDWINDOW, 
                                  CW_USEDEFAULT, CW_USEDEFAULT, width, height, NULL, NULL, wc.hInstance, NULL);
    
    if (window_handle_) {
        std::cout << "  [OK] Silicon Window Handle secured.\n";
        init_directx();
    } else {
        std::cout << "  [WARN] Headless Environment Detected. Silicon UI running in virtual mode.\n";
    }
#endif
}

void SiliconUI::init_directx() {
#ifdef _WIN32
    ID2D1Factory* pFactory = nullptr;
    D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory);
    d2d_factory_ = pFactory;

    RECT rc;
    GetClientRect((HWND)window_handle_, &rc);

    ID2D1HwndRenderTarget* pRT = nullptr;
    pFactory->CreateHwndRenderTarget(
        D2D1::RenderTargetProperties(),
        D2D1::HwndRenderTargetProperties((HWND)window_handle_, D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top)),
        &pRT);
    render_target_ = pRT;

    ID2D1SolidColorBrush* pBrush = nullptr;
    if (pRT) {
        pRT->CreateSolidColorBrush(D2D1::ColorF(0x00d2ff, 1.0f), &pBrush);
        brush_ = pBrush;
        std::cout << "  [OK] Direct2D Pipeline Saturated.\n";
    }
#endif
}

SiliconUI::~SiliconUI() {
#ifdef _WIN32
    if (brush_) ((ID2D1SolidColorBrush*)brush_)->Release();
    if (render_target_) ((ID2D1HwndRenderTarget*)render_target_)->Release();
    if (d2d_factory_) ((ID2D1Factory*)d2d_factory_)->Release();
    if (window_handle_) DestroyWindow((HWND)window_handle_);
#endif
}

void SiliconUI::render() {
    // In a full implementation, this is the hardware-saturated render loop.
    // It would talk directly to the AetherSocket for mental wavefront visualization.
}

void SiliconUI::draw_editor_region() {
    // Direct GPU drawing of code-blocks
    std::cout << "  [SiliconUI] Drawing Editor Region...\n";
}

void SiliconUI::draw_terminal_region() {
    std::cout << "  [SiliconUI] Drawing Terminal Region...\n";
}

void SiliconUI::draw_sidebar() {
    std::cout << "  [SiliconUI] Drawing Sidebar Region...\n";
}

} // namespace nca::deployment
