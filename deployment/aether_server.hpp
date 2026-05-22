#pragma once
#include <string>
#include <vector>
#include <functional>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace nca::deployment {

/**
 * AetherServer
 * High-performance, non-blocking TCP server for Silicon Telemetry.
 * Streams JSON thought-vectors and topology data to the Dashboard.
 */
class AetherServer {
public:
    AetherServer(int port = 8080);
    ~AetherServer();

    // Broadcasts telemetry to all connected dashboards
    void broadcast(const std::string& json_payload);

    // Updates the server state (Must be called in the main loop)
    void update();

private:
#ifdef _WIN32
    SOCKET listen_socket_;
    std::vector<SOCKET> clients_;
#else
    int listen_socket_;
    std::vector<int> clients_;
#endif

    void init_socket(int port);
    void handle_new_connections();
};

} // namespace nca::deployment
