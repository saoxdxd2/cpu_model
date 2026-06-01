import * as vscode from 'vscode';
import WebSocket from 'ws';

let ws: WebSocket | undefined;
let isRunning = false;

export function activate(context: vscode.ExtensionContext) {
    console.log('Aether AI Agent activated.');

    const startCmd = vscode.commands.registerCommand('aether.start', () => {
        if (isRunning) {
            vscode.window.showInformationMessage('Aether Agent is already running.');
            return;
        }
        startAgent();
    });

    const stopCmd = vscode.commands.registerCommand('aether.stop', () => {
        stopAgent();
    });

    context.subscriptions.push(startCmd, stopCmd);
    
    // Auto-start on load
    startAgent();
}

function startAgent() {
    isRunning = true;
    vscode.window.showInformationMessage('Aether Agent Booting: Connecting to Silicon Bus (Port 3001)...');

    connectWebSocket();
}

function connectWebSocket() {
    ws = new WebSocket('ws://localhost:3001');

    ws.on('open', () => {
        vscode.window.showInformationMessage('Aether Agent: Connected to Silicon Gateway.');
        
        // Register VSCode workspace to the AI Model
        ws?.send(JSON.stringify({
            type: "REGISTER_VSCODE_ENV",
            workspace: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || "no-workspace"
        }));
        
        // Continuous telemetry loop
        setInterval(() => {
            if (ws?.readyState === WebSocket.OPEN) {
                sendStateToModel();
            }
        }, 1000);
    });

    ws.on('message', (data) => {
        try {
            const msg = JSON.parse(data.toString());
            handleAgentCommand(msg);
        } catch (e) {
            console.error("Aether Message Error:", e);
        }
    });

    ws.on('close', () => {
        if (isRunning) {
            console.log('Aether Agent disconnected. Retrying in 5s...');
            setTimeout(connectWebSocket, 5000);
        }
    });
    
    ws.on('error', (err) => {
        console.error('Aether WebSocket error:', err);
    });
}

function sendStateToModel() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !ws) return;

    const payload = {
        type: "OBSERVATION",
        file: editor.document.fileName,
        cursorLine: editor.selection.active.line,
        codeContext: editor.document.getText()
    };
    
    ws.send(JSON.stringify(payload));
}

async function handleAgentCommand(msg: any) {
    // Execute actions sent by the C++ Engine
    if (msg.type === 'APPLY_EDIT') {
        const editor = vscode.window.activeTextEditor;
        if (editor && editor.document.fileName === msg.file) {
            await editor.edit(editBuilder => {
                const pos = new vscode.Position(msg.line, 0);
                editBuilder.insert(pos, msg.text + "\n");
            });
            vscode.window.showInformationMessage(`Aether injected code at line ${msg.line}`);
        }
    } else if (msg.type === 'SHOW_MESSAGE') {
        vscode.window.showInformationMessage(`[AETHER] ${msg.text}`);
    }
}

function stopAgent() {
    isRunning = false;
    if (ws) {
        ws.close();
        ws = undefined;
    }
    vscode.window.showInformationMessage('Aether Agent Stopped.');
}

export function deactivate() {
    stopAgent();
}
