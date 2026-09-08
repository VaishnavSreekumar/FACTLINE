import React, { useState } from 'react';

const STATUS_LABELS = {
  CORROBORATES: 'Corroborates',
  CONTRADICTS: 'Contradicts',
  CONTEXT_RESOLVES: 'Context Resolves',
  EVOLVES_FROM: 'Evolves From',
  SUPERSEDES: 'Supersedes',
  UNRESOLVED: 'Unresolved',
};

export default function RelationshipsView({ relationships, facts, documents }) {
  const [selectedRelId, setSelectedRelId] = useState(
    relationships.length > 0 ? relationships[0].relationship_id : null
  );
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  // Build facts lookup by fact_id
  const factsMap = React.useMemo(() => {
    const map = new Map();
    facts.forEach((f) => {
      const factId = f.fact?.fact_id || f.fact_id;
      map.set(factId, f);
    });
    return map;
  }, [facts]);

  // Build documents lookup by document_id
  const docsMap = React.useMemo(() => {
    const map = new Map();
    documents.forEach((d) => {
      map.set(d.document_id, d);
    });
    return map;
  }, [documents]);

  const filteredRelationships = relationships.filter((rel) => {
    if (statusFilter !== 'ALL' && rel.relationship_type !== statusFilter) {
      return false;
    }
    if (!searchQuery.trim()) return true;

    const q = searchQuery.toLowerCase();
    const factA = factsMap.get(rel.fact_a_id);
    const factB = factsMap.get(rel.fact_b_id);

    const entityA = factA?.canonical_entity || factA?.fact?.entity || '';
    const metricA = factA?.canonical_metric || factA?.fact?.metric || '';
    const rawValA = factA?.fact?.value_raw || '';
    const rawValB = factB?.fact?.value_raw || '';
    const explanation = rel.explanation || '';
    const type = rel.relationship_type || '';

    return (
      entityA.toLowerCase().includes(q) ||
      metricA.toLowerCase().includes(q) ||
      rawValA.toLowerCase().includes(q) ||
      rawValB.toLowerCase().includes(q) ||
      explanation.toLowerCase().includes(q) ||
      type.toLowerCase().includes(q)
    );
  });

  const selectedRel =
    relationships.find((r) => r.relationship_id === selectedRelId) ||
    filteredRelationships[0] ||
    null;

  const factA = selectedRel ? factsMap.get(selectedRel.fact_a_id) : null;
  const factB = selectedRel ? factsMap.get(selectedRel.fact_b_id) : null;

  const formatDocShort = (docId) => {
    const doc = docsMap.get(docId);
    if (doc) {
      return doc.document_name.replace(/\.pdf$/i, '');
    }
    return docId ? `Doc (${docId.substring(0, 8)})` : 'Document';
  };

  const getArrowSymbol = (type) => {
    if (type === 'EVOLVES_FROM' || type === 'SUPERSEDES') {
      return '→';
    }
    return '↔';
  };

  return (
    <div>
      {/* Filter Bar */}
      <div className="filter-bar">
        <input
          type="text"
          className="search-input"
          placeholder="Filter by entity, metric, value, or reasoning keyword..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <select
          className="select-filter"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="ALL">All Statuses ({relationships.length})</option>
          <option value="CORROBORATES">Corroborates</option>
          <option value="CONTRADICTS">Contradicts</option>
          <option value="CONTEXT_RESOLVES">Context Resolves</option>
          <option value="EVOLVES_FROM">Evolves From</option>
          <option value="SUPERSEDES">Supersedes</option>
          <option value="UNRESOLVED">Unresolved</option>
        </select>
      </div>

      {filteredRelationships.length === 0 ? (
        <div className="empty-state">
          <div className="empty-title">No Comparable Relationships Identified</div>
          <div className="empty-text">
            {relationships.length === 0
              ? 'FACTLINE did not find enough comparable claims across the uploaded documents to establish a supported relationship.'
              : 'No relationships match the selected filter criteria.'}
          </div>
        </div>
      ) : (
        <div className="split-layout">
          {/* Master Relationship List */}
          <div className="relationship-list">
            {filteredRelationships.map((rel) => {
              const fA = factsMap.get(rel.fact_a_id);
              const fB = factsMap.get(rel.fact_b_id);
              const entity = fA?.canonical_entity || fA?.fact?.entity || 'Claim';
              const metric = fA?.canonical_metric || fA?.fact?.metric || 'Metric';
              const rawA = fA?.fact?.value_raw || '—';
              const rawB = fB?.fact?.value_raw || '—';
              const period = fA?.fact?.time_period?.label || '';
              const docNameA = formatDocShort(fA?.fact?.provenance?.document_id);
              const docNameB = formatDocShort(fB?.fact?.provenance?.document_id);
              const pageA = fA?.fact?.provenance?.page_number;
              const pageB = fB?.fact?.provenance?.page_number;

              const isSelected = selectedRel && selectedRel.relationship_id === rel.relationship_id;

              return (
                <div
                  key={rel.relationship_id}
                  className={`relationship-card-row ${isSelected ? 'selected' : ''}`}
                  onClick={() => setSelectedRelId(rel.relationship_id)}
                >
                  <div className="rel-row-header">
                    <div className="rel-row-title">
                      {entity} · {metric} {period ? `(${period})` : ''}
                    </div>
                    <span className={`status-indicator ${rel.relationship_type}`}>
                      <span className="status-dot" />
                      {STATUS_LABELS[rel.relationship_type] || rel.relationship_type}
                    </span>
                  </div>

                  <div className="rel-row-values">
                    <span>{rawA}</span>
                    <span className="rel-arrow">{getArrowSymbol(rel.relationship_type)}</span>
                    <span>{rawB}</span>
                  </div>

                  <div className="rel-row-sources">
                    <span>
                      {docNameA} {pageA ? `· p.${pageA}` : ''}
                    </span>
                    <span>
                      {docNameB} {pageB ? `· p.${pageB}` : ''}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Relationship Detail Panel */}
          {selectedRel && (
            <div className="detail-panel">
              <div className="detail-header">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                  <span className={`status-indicator ${selectedRel.relationship_type}`}>
                    <span className="status-dot" />
                    {STATUS_LABELS[selectedRel.relationship_type] || selectedRel.relationship_type}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    Confidence: {(selectedRel.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="detail-title">
                  {factA?.canonical_entity || factA?.fact?.entity || 'Claim'} —{' '}
                  {factA?.canonical_metric || factA?.fact?.metric || 'Metric'}
                </div>
              </div>

              <div className="detail-body">
                {/* Visual Comparison: FACT A vs FACT B */}
                <div className="comparison-grid">
                  {/* FACT A */}
                  <div className="comparison-card">
                    <div className="comparison-card-label">FACT A</div>
                    <div className="comparison-val-raw">{factA?.fact?.value_raw || '—'}</div>
                    <div className="comparison-val-norm">
                      Normalized: {factA?.normalized_value?.value_qualifier || ''}
                      {factA?.normalized_value?.numeric_value != null
                        ? factA.normalized_value.numeric_value.toLocaleString()
                        : '—'}{' '}
                      {factA?.normalized_value?.canonical_unit || ''}
                    </div>
                    <div className="comparison-meta">
                      {formatDocShort(factA?.fact?.provenance?.document_id)} · Page {factA?.fact?.provenance?.page_number ?? '—'}
                    </div>
                  </div>

                  {/* FACT B */}
                  <div className="comparison-card">
                    <div className="comparison-card-label">FACT B</div>
                    <div className="comparison-val-raw">{factB?.fact?.value_raw || '—'}</div>
                    <div className="comparison-val-norm">
                      Normalized: {factB?.normalized_value?.value_qualifier || ''}
                      {factB?.normalized_value?.numeric_value != null
                        ? factB.normalized_value.numeric_value.toLocaleString()
                        : '—'}{' '}
                      {factB?.normalized_value?.canonical_unit || ''}
                    </div>
                    <div className="comparison-meta">
                      {formatDocShort(factB?.fact?.provenance?.document_id)} · Page {factB?.fact?.provenance?.page_number ?? '—'}
                    </div>
                  </div>
                </div>

                {/* FACTLINE Reasoning Section */}
                <div className="section-box">
                  <div className="section-label reasoning">FACTLINE REASONING</div>
                  <div className="reasoning-box">{selectedRel.explanation}</div>
                  {selectedRel.reason_codes && selectedRel.reason_codes.length > 0 && (
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
                      {selectedRel.reason_codes.map((code) => (
                        <span key={code} className="tag-badge">
                          {code}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Source Evidence Citations */}
                <div className="section-box">
                  <div className="section-label">SOURCE EVIDENCE (VERBATIM PROVENANCE)</div>

                  {/* Citation A */}
                  <div className="citation-box">
                    <div className="citation-header">
                      <span>{formatDocShort(factA?.fact?.provenance?.document_id)}</span>
                      <span className="citation-page">
                        Page {factA?.fact?.provenance?.page_number ?? '—'}
                      </span>
                    </div>
                    <div className="citation-quote">
                      "{factA?.fact?.provenance?.supporting_text || 'Exact quote unavailable'}"
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      Epistemic Status: <b>{factA?.fact?.epistemic_status || 'reported'}</b> · Period:{' '}
                      <b>{factA?.fact?.time_period?.label || 'Unspecified'}</b>
                    </div>
                  </div>

                  {/* Citation B */}
                  <div className="citation-box">
                    <div className="citation-header">
                      <span>{formatDocShort(factB?.fact?.provenance?.document_id)}</span>
                      <span className="citation-page">
                        Page {factB?.fact?.provenance?.page_number ?? '—'}
                      </span>
                    </div>
                    <div className="citation-quote">
                      "{factB?.fact?.provenance?.supporting_text || 'Exact quote unavailable'}"
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      Epistemic Status: <b>{factB?.fact?.epistemic_status || 'reported'}</b> · Period:{' '}
                      <b>{factB?.fact?.time_period?.label || 'Unspecified'}</b>
                    </div>
                  </div>
                </div>

                {/* Canonical Resolution Breakdown (Purely API-driven) */}
                <div className="section-box">
                  <div className="section-label">CANONICAL RESOLUTION BREAKDOWN</div>
                  <div className="kv-grid" style={{ background: 'var(--bg-subtle)', padding: 14, border: '1px solid var(--border-default)', borderRadius: 5 }}>
                    <div className="kv-key">Time Bounds:</div>
                    <div className="kv-val">
                      {factA?.normalized_time_period?.start_date && factA?.normalized_time_period?.end_date
                        ? `${factA.normalized_time_period.start_date} to ${factA.normalized_time_period.end_date}`
                        : factA?.fact?.time_period?.label || 'Unspecified'}
                    </div>

                    <div className="kv-key">Unit Scales:</div>
                    <div className="kv-val">
                      {factA?.normalized_value?.scale || 'Standard'} vs {factB?.normalized_value?.scale || 'Standard'} ({factA?.normalized_value?.canonical_unit || 'Units'})
                    </div>

                    {selectedRel.contextual_factors && selectedRel.contextual_factors.length > 0 && (
                      <>
                        <div className="kv-key">Evaluated Factors:</div>
                        <div className="kv-val">
                          {selectedRel.contextual_factors.join(', ')}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
