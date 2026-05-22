import React, { useState, useEffect } from 'react';
import './AetherApp.css';

// --- COMPONENTS ---
const ModelHub = () => (
  <div className="card">
    <h3>Silicon Model Hub</h3>
    <div className="model-list">
      <div className="model-item active">Gemma-4-Silicon (v27.0)</div>
      <div className="model-item locked">Mistral-7B-Adopted (Locked)</div>
      <div className="model-item locked">LLaMA-3-Saturated (Locked)</div>
    </div>
    <button className="btn-primary">Download New Foundation</button>
  </div>
);

const TopologyViewer = ({ telemetry }) => (
  <div className="card topology">
    <h3>Silicon Topology Viewer</h3>
    <div className="wiring-map">
      <div className="node active">Encoder</div>
      <div className="connector"></div>
      <div className="node pulse" style={{ opacity: 0.5 + (telemetry.wavefront || 0) * 0.5 }}>
        SDMS Expert Pool ({telemetry.active_experts || 1024})
      </div>
      <div className="connector"></div>
      <div className="node active">Spectral RLS</div>
    </div>
    <p className="telemetry">
      Wavefront Stability: {(telemetry.wavefront || 0.9842).toFixed(4)} | 
      Latency: {telemetry.latency || '0.2'} ms
    </p>
  </div>
);

// ... (registry component unchanged)

// --- MAIN APP ---
const AetherApp = () => {
  const [view, setView] = useState('dashboard');
  const [telemetry, setTelemetry] = useState({});

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:3001');
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setTelemetry(data);
      } catch (e) {
        console.error("Telemetry Parse Error", e);
      }
    };
    return () => ws.close();
  }, []);

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
              <ModelHub />
              <ToolRegistry />
            </div>
            <div className="col">
              <TopologyViewer telemetry={telemetry} />
              <div className="card training">
                <h3>Scaling & Fine-Tuning</h3>
                <div className="drop-zone">
                  DROP DATA OR DIRECTORY HERE TO TRAIN
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="ide-placeholder">
            <h2>Autonomous IDE View</h2>
            <p>The agent is currently scanning: <code>vscode/src/vs/editor/common/model.ts</code></p>
            <div className="terminal-mock">
              [BIT_STEP 500] Entropy: 1.0000 | Speed: 4.2 MB/s
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
