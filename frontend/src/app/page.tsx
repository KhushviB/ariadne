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

  // States for coordinates and annotations (fallback initialized with mock)
  const [nodes, setNodes] = useState<any[]>(mockGraphData.chromosomes['21'].nodes);
  const [edges, setEdges] = useState<any[]>(mockGraphData.chromosomes['21'].edges);
  const [annotations, setAnnotations] = useState<any>(mockGraphData.chromosomes['21'].clinical_annotations);

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

  // Fetch coordinates dynamically from remote API when available
  const [imputationResult, setImputationResult] = useState<any>(null);

  useEffect(() => {
    const rawApiUrl = process.env.NEXT_PUBLIC_API_URL || '';
    const apiBaseUrl = rawApiUrl.replace(/\/$/, ''); // Safely strip trailing slashes to avoid // double-slash 404s
    
    if (!apiBaseUrl) {
      // Use local static mock data
      const chrData = mockGraphData.chromosomes[selectedChr as '21' | '22'];
      setNodes(chrData.nodes);
      setEdges(chrData.edges);
      setAnnotations(chrData.clinical_annotations);
      return;
    }

    logMessage(`Streaming HPRC pangenome alignment sequence for Chromosome ${selectedChr}...`, 'info');
    
    fetch(`${apiBaseUrl}/api/subgraph?chr_id=${selectedChr}&cohort=${selectedCohort}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        // Map nodes
        const mappedNodes = data.nodes.map((n: any) => ({
          id: n.id,
          x: n.x,
          y: n.y,
          z: n.z,
          sequence: n.sequence || 'ACTG',
          type: n.type || 'Variant',
          frequency: n.frequency || 0.5
        }));

        // Map edges and align with attention weights
        const mappedEdges = data.edges.map((e: any, idx: number) => ({
          source: e.source,
          target: e.target,
          frequency: e.frequency || 0.5,
          attention: data.attention_weights[idx] !== undefined ? data.attention_weights[idx] : 0.1,
          cohorts: [selectedCohort]
        }));

        setNodes(mappedNodes);
        setEdges(mappedEdges);

        // Map simplified clinical annotations
        const parsedAnnots: any = {};
        for (const [key, value] of Object.entries(data.clinical_annotations)) {
          const parts = (value as string).split(' | ');
          parsedAnnots[key] = {
            rsid: parts[0] || 'Unknown',
            gene: parts[1] || 'Unknown',
            clinical_significance: parts[2] || 'VUS',
            phenotype: parts[3] || 'Unknown'
          };
        }
        setAnnotations(parsedAnnots);

        logMessage(`Chromosome ${selectedChr} map loaded. Mapped ${mappedNodes.length} variation segments.`, 'success');
      })
      .catch(err => {
        logMessage(`Cloud link failed: ${err.message}. Initializing localized baseline mapping database...`, 'warning');
        const chrData = mockGraphData.chromosomes[selectedChr as '21' | '22'];
        setNodes(chrData.nodes);
        setEdges(chrData.edges);
        setAnnotations(chrData.clinical_annotations);
      });
  }, [selectedChr, selectedCohort, logMessage]);

  // Run imputation inference
  const handleImputeNode = useCallback((nodeId: number) => {
    const rawApiUrl = process.env.NEXT_PUBLIC_API_URL || '';
    const apiBaseUrl = rawApiUrl.replace(/\/$/, '');
    
    if (!apiBaseUrl) {
      // Return static mock result
      logMessage(`Simulating clinical pathology diagnostic on segment #${nodeId}...`, 'info');
      setTimeout(() => {
        const mockResult = {
          node_id: nodeId,
          imputation_probability: 0.965,
          clinical_significance: "Likely Pathogenic",
          phenotypic_risk_score: 3.12
        };
        setImputationResult(mockResult);
        logMessage(`Diagnosis complete: Risk factor 3.12 (High susceptibility detected)`, 'success');
      }, 600);
      return;
    }

    logMessage(`Running GNN pathology diagnostics over Locus Segment #${nodeId}...`, 'info');

    fetch(`${apiBaseUrl}/api/impute?node_id=${nodeId}&chr_id=${selectedChr}`, {
      method: 'POST'
    })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setImputationResult(data);
        logMessage(`Diagnostics complete: Likelihood ${(data.imputation_probability * 100).toFixed(1)}% | PRS: ${data.phenotypic_risk_score.toFixed(2)}`, 'success');
      })
      .catch(err => {
        logMessage(`Imputation failed: ${err.message}`, 'warning');
      });
  }, [selectedChr, logMessage]);

  // Handle chromosome changes
  const handleChrChange = (chr: string) => {
    setSelectedChr(chr);
    setSelectedNodeId(null); // Clear selected node
    setImputationResult(null); // Clear imputation details
    logMessage(`Switched active view to Chromosome ${chr} genomic sequence region.`, 'info');
  };

  // Handle cohort filter changes
  const handleCohortChange = (cohort: string) => {
    setSelectedCohort(cohort);
    logMessage(`Filtering active lineage paths to target cohort: ${cohort === 'all' ? 'Global Population' : cohort}.`, 'info');
  };

  // Handle attention toggle
  const handleAttentionToggle = (show: boolean) => {
    setShowAttention(show);
    logMessage(
      show
        ? 'WebGL scene overlaid with GNN diagnostic weights.'
        : 'WebGL scene restored to standard sequence density visibility.',
      show ? 'success' : 'warning'
    );
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  // Selected node metadata and annotations
  const selectedNode = nodes.find(n => n.id === selectedNodeId) || null;
  const annotation = selectedNodeId ? (annotations as any)[`node_${selectedNodeId}`] || null : null;

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
          onImputeNode={handleImputeNode}
          imputationResult={imputationResult}
          onLogMessage={logMessage}
        />
      </main>
    </div>
  );
}
