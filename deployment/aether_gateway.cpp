#include "deployment/nca_deploy.h"
#include "deployment/peripheral_bridge.hpp"
#include "deployment/aether_server.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <thread>

using namespace nca::deployment;

/**
 * Aether Gateway
 * The central command-and-control bridge for the NCA AI IDE.
 * Translates Dashboard requests into hardware-saturated thought cycles.
 */
class AetherGateway {
public:
    AetherGateway(size_t obs_dim = 1616, size_t act_dim = 80) {
        std::cout << "[GATEWAY] Initializing Silicon Bridge...\n";
        engine_ = nca_engine_create(obs_dim, act_dim);
        peripherals_ = std::make_unique<PeripheralBridge>();
        server_ = std::make_unique<AetherServer>(8080);
    }

    ~AetherGateway() {
        nca_engine_destroy(engine_);
    }

    void run_service_step() {
        // 1. Update Server (Handle new dashboard connections)
        server_->update();

        // 2. Continuous Telemetry Broadcast
        // (In a real run, this would be triggered by actual agent steps)
        std::string telemetry = "{\"wavefront\": 0.9842, \"active_experts\": 16}";
        server_->broadcast(telemetry);
    }

    void handle_load_model(const std::string& pt_path) {
        std::cout << "[GATEWAY] Loading Model: " << pt_path << "\n";
        nca_engine_load_pt(engine_, pt_path.c_str());
    }

    std::string get_topology_report() {
        return nca_get_topology(engine_);
    }

private:
    nca_engine_ptr engine_;
    std::unique_ptr<PeripheralBridge> peripherals_;
    std::unique_ptr<AetherServer> server_;
    std::map<std::string, bool> tools_enabled_;
};

int main() {
    std::cout << "========================================================\n";
    std::cout << " NCA — AETHER GATEWAY (Silicon Control Center)        \n";
    std::cout << "========================================================\n\n";

    AetherGateway gateway;
    gateway.handle_load_model("nca_final_model.pt");

    std::cout << "[OK] Gateway Service Started. Press Ctrl+C to shutdown.\n";

    // --- NON-BLOCKING SERVICE LOOP ---
    while (true) {
        gateway.run_service_step();
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10Hz Telemetry
    }

    return 0;
}
