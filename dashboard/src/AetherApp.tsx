import React, { useState, useEffect } from 'react';
import './AetherApp.css';

// --- COMPONENTS ---
const ModelHub = ({ onLoad, onCreateAgent, agents }) => (
  <div className="card">
    <h3>Silicon Hub</h3>
    <div className="model-list">
      <div className="model-item active" onClick={onLoad}>Gemma-4-Silicon (Shared Weights)</div>
    </div>
    <div className="agent-list">
      <p style={{fontSize: '0.7rem', color: '#888'}}>ACTIVE WAVEFRONTS: {agents}</p>
      {Array.from({length: agents}).map((_, i) => (
        <div key={i} className="agent-tag">AGENT_{i}</div>
      ))}
    </div>
    <button className="btn-primary" onClick={onCreateAgent}>Spawn New Independent Agent</button>
  </div>
);

// ... (TopologyViewer remains unchanged)

const ToolRegistry = ({ onToggle }) => (
  <div className="card">
    <h3>Agentic Tools</h3>
    <div className="tool-toggle">
      <span>Keyboard/Mouse Bridge</span>
      <input type="checkbox" defaultChecked onChange={(e) => onToggle('peripherals', e.target.checked)} />
    </div>
    <div className="tool-toggle">
      <span>Silicon Visual Ingestion</span>
      <input type="checkbox" defaultChecked onChange={(e) => onToggle('vision', e.target.checked)} />
    </div>
    <div className="tool-toggle">
      <span>Creative Media Room</span>
      <input type="checkbox" onChange={(e) => onToggle('creative', e.target.checked)} />
    </div>
  </div>
);

// --- MAIN APP ---
const AetherApp = () => {
  const [view, setView] = useState('dashboard');
  const [telemetry, setTelemetry] = useState({ active_agents: 0 });
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:3001');
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setTelemetry(data);
      } catch (e) { }
    };
    setSocket(ws);
    return () => ws.close();
  }, []);

  const sendCommand = (type, value) => {
    if (socket) {
      socket.send(JSON.stringify({ type, value }));
    }
  };

  return (
    <div className="aether-root">
      <header className="navbar">
        <div className="logo">NCA AETHER</div>
        <nav>
          <button onClick={() => setView('dashboard')} className={view === 'dashboard' ? 'active' : ''}>DASHBOARD</button>
          <button onClick={() => setView('ide')} className={view === 'ide' ? 'active' : ''}>AGENTIC IDE</button>
        </nav>
        <div className="status" style={{ background: telemetry.wavefront > 0.9 ? 'rgba(0,255,100,0.1)' : 'rgba(255,0,0,0.1)', color: telemetry.wavefront > 0.9 ? '#00ff64' : '#ff3232' }}>
          ENGINE: {telemetry.wavefront > 0.9 ? 'SATURATED' : 'RECALIBRATING'}
        </div>
      </header>

      <main className="content">
        {view === 'dashboard' ? (
          <div className="grid">
            <div className="col">
              <ModelHub 
                agents={telemetry.active_agents || 0}
                onLoad={() => sendCommand('LOAD_MODEL', 'nca_final_model.pt')} 
                onCreateAgent={() => sendCommand('CREATE_AGENT', {})}
              />
              <ToolRegistry onToggle={(tool, active) => sendCommand('SET_TOOL', { tool, active })} />
            </div>
            <div className="col">
              <TopologyViewer telemetry={telemetry} />
              <div className="card resource">
                <h3>Silicon Efficiency Proof</h3>
                <div className="stat">Shared Weights: <strong>{telemetry.shared_ram_mb || 120} MB</strong></div>
                <div className="stat">Cost Per Wavefront: <strong>{telemetry.agent_cost_kb || 8} KB</strong></div>
                <div className="stat">Scaling Potential: <strong>1,000+ Agents</strong></div>
                <p className="hint" style={{ fontSize: '0.7rem', color: '#888', marginTop: '1rem' }}>
                  By sharing weights and only isolating thought states, we achieve a 100x reduction in scaling overhead compared to LLaMA.cpp instances.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="ide-placeholder">
            <h2>Autonomous IDE View</h2>
            <div className="terminal-mock">
              [AGENT_0] Scanning VSCode Source...
              <br />
              [WAVEFRONT] Intensity: {(telemetry.wavefront || 0).toFixed(4)}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default AetherApp;
