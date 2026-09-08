import React from 'react';

export default function AnalysisProgress({ fileCount }) {
  return (
    <div className="progress-box">
      <div className="spinner" />
      <div className="progress-title">
        Analyzing {fileCount} {fileCount === 1 ? 'Document' : 'Documents'}…
      </div>
      <div className="progress-subtitle">
        Parsing page text, extracting evidence-grounded facts, normalizing dimensions, and evaluating cross-document claim comparability.
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 14 }}>
        This may take a moment for multi-page documents.
      </div>
    </div>
  );
}
