#pragma once
#include <string>

#include "deployment/nca_deploy.h"

namespace nca::deployment {

/**
 * SiliconUI
 * Lightweight C++ native windowing system to replace Electron.
 * Uses Direct2D/DirectX (Windows) or Vulkan for hardware saturation.
 */
class NCA_API SiliconUI {
public:
     SiliconUI(const std::string& title, int width = 1280, int height = 720);
    ~SiliconUI();

    // The Render Loop: Bypasses the heavy browser shell
    void render();

    // Native Component Primitives
    void draw_editor_region();
    void draw_terminal_region();
    void draw_sidebar();

private:
    void* window_handle_;
    
    // [HARDWARE SATURATION] Direct2D Pipeline
    void* d2d_factory_;
    void* render_target_;
    void* brush_;

    void init_directx();
};

} // namespace nca::deployment
