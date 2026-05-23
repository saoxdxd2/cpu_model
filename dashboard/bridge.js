const net = require('net');
const WebSocket = require('ws');

/**
 * Aether Bridge
 * Connects the C++ Silicon Engine (TCP:8080) to the React Dashboard (WS:3000)
 */

const TCP_PORT = 8080;
const WS_PORT = 3001;

console.log("===========================================");
console.log(" NCA Aether Bridge: Silicon-to-Web        ");
console.log("===========================================");

const wss = new WebSocket.Server({ port: WS_PORT });
let cppSocket = null;

// 1. Establish Connection to C++ Engine
function connectToEngine() {
    cppSocket = new net.Socket();
    
    cppSocket.connect(TCP_PORT, '127.0.0.1', () => {
        console.log(`[BRIDGE] Connected to Aether Gateway (TCP:${TCP_PORT})`);
    });

    cppSocket.on('data', (data) => {
        // Broadcast raw silicon telemetry to all dashboard clients
        const payload = data.toString();
        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(payload);
            }
        });
    });

    cppSocket.on('error', (err) => {
        console.log(`[BRIDGE] Engine Link Error: ${err.message}. Retrying in 5s...`);
        setTimeout(connectToEngine, 5000);
    });
}

// 2. Handle Dashboard Commands
wss.on('connection', (ws) => {
    console.log("[BRIDGE] New Dashboard GUI connected.");
    
    ws.on('message', (message) => {
        try {
            const cmd = JSON.parse(message);
            console.log(`[BRIDGE] Uplink: ${cmd.type} -> Silicon`);
            
            // Forward commands to C++ (AetherGateway)
            if (cppSocket && cppSocket.writable) {
                // Wrap in JSON-RPC style for the C++ side
                const payload = JSON.stringify(cmd) + "\n";
                cppSocket.write(payload);
            }
        } catch (e) {
            console.error("[BRIDGE] Malformed command from Dashboard:", message.toString());
        }
    });
});

connectToEngine();
console.log(`[BRIDGE] WebSocket Mediator listening on WS:${WS_PORT}`);
