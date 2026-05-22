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

const TopologyViewer = () => (
  <div className="card topology">
    <h3>Silicon Topology Viewer</h3>
    <div className="wiring-map">
      <div className="node active">Encoder</div>
      <div className="connector"></div>
      <div className="node pulse">SDMS Expert Pool (1024)</div>
      <div className="connector"></div>
      <div className="node active">Spectral RLS</div>
    </div>
    <p className="telemetry">Wavefront Stability: 0.9842 | Entropy: 2.14</p>
  </div>
);

const ToolRegistry = () => (
  <div className="card">
    <h3>Tool & Peripheral Registry</h3>
    <div className="tool-toggle">
      <span>Keyboard Emulation</span>
      <input type="checkbox" checked readOnly />
    </div>
    <div className="tool-toggle">
      <span>Mouse Control</span>
      <input type="checkbox" checked readOnly />
    </div>
    <div className="tool-toggle">
      <span>Visual Screenshotting</span>
      <input type="checkbox" checked readOnly />
    </div>
    <div className="tool-toggle">
      <span>AI Image Room</span>
      <input type="checkbox" />
    </div>
  </div>
);

// --- MAIN APP ---
const AetherApp = () => {
  const [view, setView] = useState('dashboard'); // 'dashboard' or 'ide'

  return (
    <div className="aether-root">
      <header className="navbar">
        <div className="logo">NCA AETHER</div>
        <nav>
          <button onClick={() => setView('dashboard')} className={view === 'dashboard' ? 'active' : ''}>DASHBOARD</button>
          <button onClick={() => setView('ide')} className={view === 'ide' ? 'active' : ''}>AGENTIC IDE</button>
        </nav>
        <div className="status">ENGINE: SATURATED</div>
      </header>

      <main className="content">
        {view === 'dashboard' ? (
          <div className="grid">
            <div className="col">
              <ModelHub />
              <ToolRegistry />
            </div>
            <div className="col">
              <TopologyViewer />
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
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default AetherApp;
