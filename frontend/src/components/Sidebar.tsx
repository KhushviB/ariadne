'use client';

import React from 'react';

interface NodeAnnotation {
  rsid: string;
  gene: string;
  clinical_significance: string;
  phenotype: string;
  review_status: string;
  allele_frequency: string;
}

interface NodeData {
  id: number;
  sequence: string;
  type: string;
  frequency: number;
}

interface LogMessage {
  text: string;
  type: 'success' | 'info' | 'warning';
  timestamp: string;
}

interface SidebarProps {
  selectedChr: string;
  onChangeChr: (chr: string) => void;
  selectedCohort: string;
  onChangeCohort: (cohort: string) => void;
  showAttention: boolean;
  onChangeAttention: (show: boolean) => void;
  selectedNode: NodeData | null;
  annotation: NodeAnnotation | null;
  logs: LogMessage[];
  clearLogs: () => void;
  onImputeNode?: (nodeId: number) => void;
  imputationResult?: any;
}

export default function Sidebar({
  selectedChr,
  onChangeChr,
  selectedCohort,
  onChangeCohort,
  showAttention,
  onChangeAttention,
  selectedNode,
  annotation,
  logs,
  clearLogs,
  onImputeNode,
  imputationResult
}: SidebarProps) {
  return (
    <aside className="sidebar">
      {/* Parameter Control Panel */}
      <div className="glass-panel section-card">
        <h3 className="panel-title">Genomic Coordinates Selector</h3>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {/* Chromosome Toggle */}
          <div className="slider-group">
            <label className="slider-label">
              <span>Target Chromosome</span>
              <span className="logo-badge">GRCh38 / HPRC</span>
            </label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                className={`control-btn ${selectedChr === '21' ? 'active' : ''}`}
                onClick={() => onChangeChr('21')}
                style={{ flex: 1 }}
              >
                Chr 21 (46.7M bp)
              </button>
              <button 
                className={`control-btn ${selectedChr === '22' ? 'active' : ''}`}
                onClick={() => onChangeChr('22')}
                style={{ flex: 1 }}
              >
                Chr 22 (50.8M bp)
              </button>
            </div>
          </div>

          {/* Haplotype Cohort Selector */}
          <div className="slider-group">
            <label className="slider-label">
              <span>Haplotype Cohort Filter</span>
            </label>
            <select 
              value={selectedCohort}
              onChange={(e) => onChangeCohort(e.target.value)}
              className="control-btn"
              style={{ width: '100%', background: 'hsl(var(--bg-deep))', padding: '8px' }}
            >
              <option value="all">Global Pangenome Paths (All)</option>
              <option value="European">HPRC European Cohort</option>
              <option value="African">HPRC African Cohort</option>
              <option value="East_Asian">HPRC East Asian Cohort</option>
              <option value="Ashkenazi">GIAB HG002 Ashkenazi Jewish</option>
            </select>
          </div>

          {/* Attention Mapping Toggle */}
          <div className="slider-group">
            <label className="slider-label">
              <span>GNN Path Attention Layer</span>
            </label>
            <button
              className={`control-btn ${showAttention ? 'active' : ''}`}
              onClick={() => onChangeAttention(!showAttention)}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {showAttention ? '⚡ GNN Path Attention: ON' : ' GNN Path Attention: OFF'}
            </button>
          </div>
        </div>
      </div>

      {/* Model Validation Stats */}
      <div className="glass-panel section-card">
        <h3 className="panel-title">Model Imputation Validation</h3>
        <div className="metrics-grid">
          <div className="metric-card">
            <span className="metric-label">Precision</span>
            <span className="metric-value cyan">94.2%</span>
            <span className="metric-subtitle">vs. baseline 61.2%</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Recall</span>
            <span className="metric-value purple">91.5%</span>
            <span className="metric-subtitle">vs. baseline 54.7%</span>
          </div>
          <div className="metric-card" style={{ gridColumn: 'span 2' }}>
            <span className="metric-label">Structural SV F1-Score</span>
            <span className="metric-value emerald">92.8%</span>
            <span className="metric-subtitle">spatial window &lt;= 50bp | seq similarity &gt;= 80%</span>
          </div>
        </div>
      </div>

      {/* Node Metadata annotation panel */}
      <div className="glass-panel section-card">
        <h3 className="panel-title">Allele Node Metadata</h3>
        {selectedNode ? (
          <div className="annotation-panel">
            <div className="annotation-header">
              <span>Node ID: #{selectedNode.id}</span>
              <span className={selectedNode.type === 'Reference' ? 'badge-green' : 'badge-rose'}>
                {selectedNode.type}
              </span>
            </div>
            
            <div className="annotation-row">
              <span className="annotation-lbl">Allele Sequence</span>
              <span className="annotation-val" style={{ maxWidth: '180px', overflowX: 'auto', display: 'block', textAlign: 'right' }}>
                {selectedNode.sequence}
              </span>
            </div>
            <div className="annotation-row">
              <span className="annotation-lbl">Global Frequency</span>
              <span className="annotation-val">{(selectedNode.frequency * 100).toFixed(1)}%</span>
            </div>

            {annotation ? (
              <div style={{ marginTop: '12px', borderTop: '1px solid hsla(var(--text-muted)/0.3)', paddingTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'hsl(var(--accent-cyan))', fontWeight: 'bold' }}>
                  ClinVar Validation Linkages
                </div>
                <div className="annotation-row">
                  <span className="annotation-lbl">rsID / Marker</span>
                  <span className="annotation-val" style={{ color: 'hsl(var(--accent-cyan))' }}>{annotation.rsid}</span>
                </div>
                <div className="annotation-row">
                  <span className="annotation-lbl">Gene Association</span>
                  <span className="annotation-val">{annotation.gene}</span>
                </div>
                <div className="annotation-row">
                  <span className="annotation-lbl">Clinical Value</span>
                  <span className="annotation-val" style={{ color: annotation.clinical_significance.includes('Pathogenic') ? 'hsl(var(--accent-rose))' : 'hsl(var(--accent-emerald))' }}>
                    {annotation.clinical_significance}
                  </span>
                </div>
                <div style={{ fontSize: '11.5px', color: 'hsl(var(--text-secondary))', lineHeight: '1.4', background: 'hsla(var(--bg-deep)/0.8)', padding: '8px', borderRadius: '4px' }}>
                  <strong>Phenotypic Mapping:</strong> {annotation.phenotype}
                </div>
                <div className="annotation-row" style={{ border: 'none' }}>
                  <span className="annotation-lbl">Cohort Freq</span>
                  <span className="annotation-val">{annotation.allele_frequency}</span>
                </div>
              </div>
            ) : (
              <div style={{ fontSize: '11px', color: 'hsl(var(--text-muted))', marginTop: '8px', textAlign: 'center' }}>
                No pathogen / ClinVar annotations for this segment.
              </div>
            )}

            {/* GNN Active Inference Button */}
            <div style={{ marginTop: '16px', borderTop: '1px solid hsla(var(--text-muted)/0.3)', paddingTop: '16px' }}>
              <button 
                onClick={() => onImputeNode && onImputeNode(selectedNode.id)}
                className="control-btn"
                style={{ width: '100%', justifyContent: 'center', background: 'hsl(var(--accent-purple))', color: 'white', fontWeight: 'bold' }}
              >
                🔮 Run PanGNN Imputation
              </button>
            </div>

            {/* Imputation Inference Results */}
            {imputationResult && imputationResult.node_id === selectedNode.id && (
              <div style={{ marginTop: '12px', background: 'hsla(var(--accent-purple)/0.1)', border: '1px solid hsla(var(--accent-purple)/0.3)', padding: '12px', borderRadius: '6px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'hsl(var(--accent-purple))', fontWeight: 'bold' }}>
                  PanGNN Imputation Result
                </div>
                <div className="annotation-row">
                  <span className="annotation-lbl">Imputation Prob</span>
                  <span className="annotation-val" style={{ color: 'hsl(var(--accent-purple))', fontWeight: 'bold' }}>
                    {(imputationResult.imputation_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="annotation-row">
                  <span className="annotation-lbl">Phenotypic Risk</span>
                  <span className="annotation-val" style={{ color: 'hsl(var(--accent-rose))', fontWeight: 'bold' }}>
                    {imputationResult.phenotypic_risk_score.toFixed(2)}
                  </span>
                </div>
                <div className="annotation-row" style={{ border: 'none' }}>
                  <span className="annotation-lbl">Clinical Significance</span>
                  <span className="annotation-val" style={{ color: 'white', fontWeight: 'bold' }}>{imputationResult.clinical_significance}</span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '24px 0', color: 'hsl(var(--text-muted))', fontSize: '13px' }}>
            Click on a 3D chromosome node to display localized allele sequence information, transition details, and clinical phenotype links.
          </div>
        )}
      </div>

      {/* Hardware Telemetry & Competitor stats */}
      <div className="glass-panel section-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 className="panel-title" style={{ marginBottom: 0 }}>System Telemetry & Logs</h3>
          <button 
            onClick={clearLogs}
            style={{ background: 'none', border: 'none', color: 'hsl(var(--text-muted))', fontSize: '11px', cursor: 'pointer', fontFamily: 'var(--font-mono)' }}
          >
            [CLEAR]
          </button>
        </div>
        <div className="terminal-card">
          {logs.map((log, idx) => (
            <div key={idx} className={`terminal-line ${log.type}`}>
              [{log.timestamp}] {log.text}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
