'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Visualizer3D from '../components/Visualizer3D';
import Sidebar from '../components/Sidebar';
import mockGraphData from '../../public/mock-data/mockGraph.json';

interface LogMessage {
  text: string;
  type: 'success' | 'info' | 'warning';
  timestamp: string;
}

export default function Dashboard() {
  const [selectedChr, setSelectedChr] = useState<string>('21');
  const [selectedCohort, setSelectedCohort] = useState<string>('all');
  const [showAttention, setShowAttention] = useState<boolean>(false);
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [logs, setLogs] = useState<LogMessage[]>([]);

  // Function to add diagnostic logs with timestamp
  const logMessage = useCallback((text: string, type: 'success' | 'info' | 'warning' = 'info') => {
    const now = new Date();
    const timestamp = now.toTimeString().split(' ')[0];
    setLogs(prev => [{ text, type, timestamp }, ...prev].slice(0, 100)); // limit to 100 items
  }, []);

  // Initialize with greeting logs
  useEffect(() => {
    logMessage('PanGNN Foundation Dashboard initialized.', 'success');
    logMessage('System ready. Downsampled Chromosomes 21 and 22 loaded.', 'info');
  }, [logMessage]);

  // Handle chromosome changes
  const handleChrChange = (chr: string) => {
    setSelectedChr(chr);
    setSelectedNodeId(null); // Clear selected node
    logMessage(`Switched coordinate mapping viewport to Chromosome ${chr}.`, 'info');
  };

  // Handle cohort filter changes
  const handleCohortChange = (cohort: string) => {
    setSelectedCohort(cohort);
    logMessage(`Filtering topological paths to Cohort: ${cohort === 'all' ? 'All (Global)' : cohort}.`, 'info');
  };

  // Handle attention toggle
  const handleAttentionToggle = (show: boolean) => {
    setShowAttention(show);
    logMessage(
      show
        ? 'Path-Aware GNN Attention weights rendered in WebGL space.'
        : 'Restored uniform population-frequency edge visibility.',
      show ? 'success' : 'warning'
    );
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  // Load current chromosome data
  const chrData = mockGraphData.chromosomes[selectedChr as '21' | '22'];
  const nodes = chrData.nodes;
  const edges = chrData.edges;

  // Selected node metadata and annotations
  const selectedNode = nodes.find(n => n.id === selectedNodeId) || null;
  const annotation = selectedNodeId ? (chrData.clinical_annotations as any)[`node_${selectedNodeId}`] || null : null;

  return (
    <div className="dashboard-container">
      {/* Top Navigation / Header */}
      <header className="header">
        <div className="logo-section">
          <div className="logo-icon" />
          <h1 className="logo-text">PanGNN</h1>
          <span className="logo-badge">V1.0.0-BETA</span>
        </div>
        <div className="header-info" style={{ display: 'flex', gap: '16px', fontSize: '13px', color: 'hsl(var(--text-secondary))' }}>
          <span><strong>Thesis Mode:</strong> Structural Variant Imputation & Phenotypic Mapping</span>
          <span>•</span>
          <span><strong>Compute Engine:</strong> Qdrant + PyG (Remote Connection Setup)</span>
        </div>
      </header>

      {/* Primary Split Viewport */}
      <main className="viewport-layout">
        {/* Left Side: 3D WebGL Tube Map */}
        <Visualizer3D
          nodes={nodes}
          edges={edges}
          selectedNodeId={selectedNodeId}
          onSelectNode={setSelectedNodeId}
          selectedCohort={selectedCohort}
          showAttention={showAttention}
          onLogMessage={logMessage}
        />

        {/* Right Side: Analysis Panels */}
        <Sidebar
          selectedChr={selectedChr}
          onChangeChr={handleChrChange}
          selectedCohort={selectedCohort}
          onChangeCohort={handleCohortChange}
          showAttention={showAttention}
          onChangeAttention={handleAttentionToggle}
          selectedNode={selectedNode}
          annotation={annotation}
          logs={logs}
          clearLogs={handleClearLogs}
        />
      </main>
    </div>
  );
}
