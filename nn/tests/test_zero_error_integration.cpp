/**
 * NCA — Zero-Error Integration Stress Test
 * Runs the 'Silicon Swarm', 'Indexer', and 'Direct2D Primitives' 
 * in a high-speed concurrent loop to ensure absolute stability.
 */

#include "core/execution/multimodal_engine.hpp"
#include "encoding/silicon_indexer.hpp"
#include "deployment/silicon_ui.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

using namespace nca::execution;
using namespace nca::encoding;
using namespace nca::deployment;

int main() {
    std::cout << "========================================================\n";
    std::cout << " NCA — ZERO-ERROR INTEGRATION STRESS TEST             \n";
    std::cout << "========================================================\n\n";

    const size_t D = 2048;
    auto engine = std::make_shared<MultimodalEngine>(1616, 80);
    SiliconIndexer indexer("../vscode/src");
    
    std::atomic<bool> running{true};
    std::atomic<int> errors{0};

    // 1. Thread A: Continuous Background Indexing
    std::thread indexer_thread([&]() {
        std::cout << "[Thread 1] Starting Background Indexer...\n";
        try { indexer.index_all(); } catch(...) { errors++; }
    });

    // 2. Thread B: High-Frequency Swarm Reasoning
    std::thread reasoning_thread([&]() {
        std::cout << "[Thread 2] Starting Swarm Wavefront Cycles...\n";
        float input[2048] = {0.5f};
        float output[128 * 81];
        for(int i=0; i<50; ++i) {
            try { engine->step_swarm(input, output, 12); } catch(...) { errors++; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // 3. Main Thread: UI Primitive Verification
    std::cout << "[Thread 3] Stressing Native UI Primitives...\n";
    try {
        SiliconUI ui("Stress Test View");
        for(int i=0; i<20; ++i) {
            ui.render();
            ui.draw_editor_region();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } catch(...) { errors++; }

    // Wait for stress threads
    reasoning_thread.join();
    indexer_thread.join();

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << "  STRESS RESULTS: " << errors << " Errors Detected.\n";
    std::cout << "--------------------------------------------------------\n";

    if (errors == 0) {
        std::cout << "[SUCCESS] ZERO-ERROR STABILITY PROVEN.\n";
        return 0;
    } else {
        std::cout << "[FAILURE] STABILITY BREACHED. INVESTIGATE LOGS.\n";
        return 1;
    }
}
