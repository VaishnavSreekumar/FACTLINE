import React, { useState } from 'react';

export default function FactsView({ facts, documents }) {
  const [selectedFactId, setSelectedFactId] = useState(
    facts.length > 0 ? (facts[0].fact?.fact_id || facts[0].fact_id) : null
  );
  const [searchQuery, setSearchQuery] = useState('');
  const [epistemicFilter, setEpistemicFilter] = useState('ALL');

  const docsMap = React.useMemo(() => {
    const map = new Map();
    documents.forEach((d) => map.set(d.document_id, d));
    return map;
  }, [documents]);

  const filteredFacts = facts.filter((f) => {
    const factObj = f.fact || f;
    const status = factObj.epistemic_status || 'reported';
    if (epistemicFilter !== 'ALL' && status.toLowerCase() !== epistemicFilter.toLowerCase()) {
      return false;
    }
    if (!searchQuery.trim()) return true;

    const q = searchQuery.toLowerCase();
    const entity = (f.canonical_entity || factObj.entity || '').toLowerCase();
    const metric = (f.canonical_metric || factObj.metric || '').toLowerCase();
    const valRaw = (factObj.value_raw || '').toLowerCase();
    const provText = (factObj.provenance?.supporting_text || '').toLowerCase();

    return (
      entity.includes(q) ||
      metric.includes(q) ||
      valRaw.includes(q) ||
      provText.includes(q)
    );
  });

  const selectedFact =
    facts.find((f) => (f.fact?.fact_id || f.fact_id) === selectedFactId) ||
    filteredFacts[0] ||
    null;

  const currentFactObj = selectedFact?.fact || selectedFact;
  const currentNormVal = selectedFact?.normalized_value;
  const currentDoc = currentFactObj?.provenance?.document_id
    ? docsMap.get(currentFactObj.provenance.document_id)
    : null;

  return (
    <div>
      {/* Filter Bar */}
      <div className="filter-bar">
        <input
          type="text"
          className="search-input"
          placeholder="Filter facts by entity, metric, or value..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <select
          className="select-filter"
          value={epistemicFilter}
          onChange={(e) => setEpistemicFilter(e.target.value)}
        >
          <option value="ALL">All Epistemic Statuses</option>
          <option value="reported">Reported</option>
          <option value="audited">Audited</option>
          <option value="estimated">Estimated</option>
          <option value="projected">Projected</option>
          <option value="target">Target</option>
        </select>
      </div>

      {filteredFacts.length === 0 ? (
        <div className="empty-state">
          <div className="empty-title">No Facts Found</div>
          <div className="empty-text">No facts match your current search query.</div>
        </div>
      ) : (
        <div className="split-layout">
          {/* Master Table */}
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Entity</th>
                  <th>Metric</th>
                  <th>Value</th>
                  <th>Period</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredFacts.map((f) => {
                  const fact = f.fact || f;
                  const factId = fact.fact_id;
                  const isSelected = selectedFact && currentFactObj?.fact_id === factId;

                  return (
                    <tr
                      key={factId}
                      className={isSelected ? 'selected' : ''}
                      onClick={() => setSelectedFactId(factId)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td style={{ fontWeight: 600 }}>
                        {f.canonical_entity || fact.entity}
                      </td>
                      <td>{f.canonical_metric || fact.metric}</td>
                      <td style={{ fontWeight: 700, fontFeatureSettings: "'tnum' 1" }}>
                        {fact.value_raw}
                      </td>
                      <td>{fact.time_period?.label || '—'}</td>
                      <td>
                        <span className="tag-badge">{fact.epistemic_status}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Fact Detail Pane */}
          {selectedFact && (
            <div className="detail-panel">
              <div className="detail-header">
                <div>
                  <span className="tag-badge" style={{ marginBottom: 6 }}>
                    {currentFactObj.epistemic_status?.toUpperCase()}
                  </span>
                  <div className="detail-title">
                    {selectedFact.canonical_entity || currentFactObj.entity} —{' '}
                    {selectedFact.canonical_metric || currentFactObj.metric}
                  </div>
                </div>
              </div>

              <div className="detail-body">
                {/* Reported Value Card */}
                <div className="comparison-card">
                  <div className="comparison-card-label">REPORTED CLAIM</div>
                  <div className="comparison-val-raw">{currentFactObj.value_raw}</div>
                  <div className="comparison-val-norm">
                    Normalized: {currentNormVal?.value_qualifier || ''}
                    {currentNormVal?.numeric_value != null
                      ? currentNormVal.numeric_value.toLocaleString()
                      : '—'}{' '}
                    {currentNormVal?.canonical_unit || currentFactObj.unit || ''}
                  </div>
                  <div className="comparison-meta">
                    Period: {currentFactObj.time_period?.label || 'Unspecified'} · Scope:{' '}
                    {currentFactObj.scope || 'General'}
                  </div>
                </div>

                {/* Source Provenance Citation */}
                <div className="section-box">
                  <div className="section-label">SOURCE EVIDENCE (RESEARCH CITATION)</div>
                  <div className="citation-box">
                    <div className="citation-header">
                      <span>
                        {currentDoc
                          ? currentDoc.document_name.replace(/\.pdf$/i, '')
                          : 'Document'}
                      </span>
                      <span className="citation-page">
                        Page {currentFactObj.provenance?.page_number ?? '—'}
                      </span>
                    </div>
                    <div className="citation-quote">
                      "{currentFactObj.provenance?.supporting_text || 'Exact quote unavailable'}"
                    </div>
                  </div>
                </div>

                {/* Transparent Normalization Table */}
                <div className="section-box">
                  <div className="section-label">TRANSPARENT NORMALIZATION SPECIFICATION</div>
                  <div
                    className="kv-grid"
                    style={{
                      background: 'var(--bg-subtle)',
                      padding: 14,
                      border: '1px solid var(--border-default)',
                      borderRadius: 5,
                    }}
                  >
                    <div className="kv-key">Canonical Numeric:</div>
                    <div className="kv-val">
                      {currentNormVal?.numeric_value != null
                        ? currentNormVal.numeric_value.toLocaleString()
                        : '—'}
                    </div>

                    <div className="kv-key">Unit & Scale:</div>
                    <div className="kv-val">
                      {currentNormVal?.canonical_unit || currentFactObj.unit || '—'} (
                      {currentNormVal?.scale || 'Standard scale'})
                    </div>

                    <div className="kv-key">Bounded Period:</div>
                    <div className="kv-val">
                      {selectedFact.normalized_time_period?.start_date &&
                      selectedFact.normalized_time_period?.end_date
                        ? `${selectedFact.normalized_time_period.start_date} to ${selectedFact.normalized_time_period.end_date}`
                        : currentFactObj.time_period?.label || 'Unspecified'}
                    </div>

                    <div className="kv-key">Geography:</div>
                    <div className="kv-val">
                      {currentFactObj.geography || 'Unspecified'}
                    </div>

                    <div className="kv-key">Data Vintage:</div>
                    <div className="kv-val">
                      {currentFactObj.data_vintage || 'Unspecified'}
                    </div>

                    <div className="kv-key">Extraction Confidence:</div>
                    <div className="kv-val">
                      {(currentFactObj.extraction_confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                {selectedFact.normalization_warnings &&
                  selectedFact.normalization_warnings.length > 0 && (
                    <div className="section-box">
                      <div
                        className="section-label"
                        style={{ color: 'var(--status-supersedes-text)' }}
                      >
                        AUDIT & NORMALIZATION WARNINGS
                      </div>
                      <ul
                        style={{
                          fontSize: 12,
                          paddingLeft: 16,
                          color: 'var(--text-secondary)',
                        }}
                      >
                        {selectedFact.normalization_warnings.map((w, idx) => (
                          <li key={idx}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
