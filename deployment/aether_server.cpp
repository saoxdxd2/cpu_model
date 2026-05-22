#include "deployment/aether_server.hpp"
#include <iostream>
#include <fcntl.h>

namespace nca::deployment {

AetherServer::AetherServer(int port) {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    init_socket(port);
}

AetherServer::~AetherServer() {
#ifdef _WIN32
    closesocket(listen_socket_);
    for(auto s : clients_) closesocket(s);
    WSACleanup();
#else
    close(listen_socket_);
    for(auto s : clients_) close(s);
#endif
}

void AetherServer::init_socket(int port) {
    listen_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<u_short>(port));

    bind(listen_socket_, (sockaddr*)&addr, sizeof(addr));
    listen(listen_socket_, 5);

    // Set to non-blocking
#ifdef _WIN32
    u_long mode = 1;
    ioctlsocket(listen_socket_, FIONBIO, &mode);
#else
    fcntl(listen_socket_, F_SETFL, O_NONBLOCK);
#endif

    std::cout << "[SERVER] Aether Silicon Streamer listening on port " << port << "\n";
}

void AetherServer::broadcast(const std::string& json_payload) {
    std::string msg = json_payload + "\n";
    auto it = clients_.begin();
    while (it != clients_.end()) {
        int res = send(*it, msg.c_str(), static_cast<int>(msg.length()), 0);
        if (res <= 0) {
            std::cout << "[SERVER] Dashboard Disconnected.\n";
            it = clients_.erase(it);
        } else {
            ++it;
        }
    }
}

void AetherServer::update() {
    handle_new_connections();
    poll_clients();
}

void AetherServer::poll_clients() {
    char buffer[1024];
    for (auto s : clients_) {
        int res = recv(s, buffer, sizeof(buffer) - 1, 0);
        if (res > 0) {
            buffer[res] = '\0';
            command_queue_.push_back(std::string(buffer));
        }
    }
}

bool AetherServer::get_next_command(std::string& cmd_out) {
    if (command_queue_.empty()) return false;
    cmd_out = command_queue_.front();
    command_queue_.erase(command_queue_.begin());
    return true;
}

void AetherServer::handle_new_connections() {
    sockaddr_in client_addr;
    int addr_len = sizeof(client_addr);
#ifdef _WIN32
    SOCKET client = accept(listen_socket_, (sockaddr*)&client_addr, &addr_len);
    if (client != INVALID_SOCKET) {
        std::cout << "[SERVER] New Dashboard Connection established.\n";
        clients_.push_back(client);
    }
#else
    int client = accept(listen_socket_, (sockaddr*)&client_addr, (socklen_t*)&addr_len);
    if (client > 0) {
        std::cout << "[SERVER] New Dashboard Connection established.\n";
        clients_.push_back(client);
    }
#endif
}

} // namespace nca::deployment
