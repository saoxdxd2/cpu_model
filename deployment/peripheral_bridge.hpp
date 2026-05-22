#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace nca::deployment {

/**
 * PeripheralBridge
 * Hard-wires the AI model to physical OS inputs.
 * Maps 80-dim Action Vectors to Keyboard, Mouse, and Media triggers.
 */
class PeripheralBridge {
public:
    PeripheralBridge();

    // Executes a complex agentic action based on model intent
    void execute_agentic_action(const float* action_vector);

    // Peripheral Primitives
    void tap_key(uint8_t key_code);
    void move_mouse(float rel_x, float rel_y);
    void trigger_click(int button); // 0=Left, 1=Right
    
    // Media Triggers
    void capture_screen(const std::string& save_path);
    void trigger_ai_image_gen(const std::string& prompt);

private:
    // OS-level hooks (Windows/Linux/MacOS)
    void os_send_input(int type, int code, float value);
};

} // namespace nca::deployment
