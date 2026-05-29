import React from 'react';
import ReactDOM from 'react-dom/client';
import './AetherApp.css';
import AetherApp from './AetherApp';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <AetherApp />
  </React.StrictMode>
);
