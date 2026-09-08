import React from 'react';

export default function Header({ currentAnalysis, onNewAnalysis }) {
  const docCount = currentAnalysis?.documents?.length || 0;
  const factCount = currentAnalysis?.facts?.length || 0;
  const relCount = currentAnalysis?.relationships?.length || 0;

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-mark">
          FACTLINE
        </div>
        <div className="brand-divider" />
        <div className="brand-tagline">
          Evidence-First Cross-Document Fact Intelligence
        </div>
      </div>

      <div className="header-actions">
        {currentAnalysis && (
          <>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginRight: 6, fontFeatureSettings: "'tnum' 1" }}>
              {docCount} {docCount === 1 ? 'document' : 'documents'} · {factCount} {factCount === 1 ? 'fact' : 'facts'} · {relCount} {relCount === 1 ? 'relationship' : 'relationships'}
            </div>
            <button className="btn btn-sm" onClick={onNewAnalysis}>
              + New Analysis
            </button>
          </>
        )}
      </div>
    </header>
  );
}
