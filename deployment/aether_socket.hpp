#pragma once
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include "deployment/nca_deploy.h"

namespace nca::deployment {

/**
 * AetherSocket (Silicon Bus)
 * High-performance C++ WebSocket implementation to replace Node.js bridge.
 * Unified communication for Dashboard, IDE, and Engine.
 */
class NCA_API AetherSocket {
public:
    AetherSocket(int port = 3001);
    ~AetherSocket();

    // Starts the non-blocking event loop
    void start();
    void stop();

    // Broadcasts silicon telemetry directly to the browser
    void broadcast(const std::string& json_payload);

    // Processes incoming GUI commands (JSON-RPC)
    bool get_next_command(std::string& cmd_out);

private:
    int port_;
    bool running_;
    std::vector<std::string> command_queue_;
    std::mutex queue_mutex_;

    // Native OS hooks for WebSocket (Simplified for the v1.4 Proof)
    void handle_network_io();
};

} // namespace nca::deployment
