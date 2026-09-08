import React from 'react';

export default function DocumentsView({ documents, facts, extractionStatus }) {
  // Count facts per document
  const factsPerDoc = React.useMemo(() => {
    const counts = new Map();
    facts.forEach((f) => {
      const fact = f.fact || f;
      const docId = fact.provenance?.document_id;
      if (docId) {
        counts.set(docId, (counts.get(docId) || 0) + 1);
      }
    });
    return counts;
  }, [facts]);

  // Map doc extraction status if available
  const docStatusMap = React.useMemo(() => {
    const map = new Map();
    if (extractionStatus?.documents) {
      extractionStatus.documents.forEach((d) => {
        map.set(d.document_id, d);
      });
    }
    return map;
  }, [extractionStatus]);

  if (!documents || documents.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-title">No Documents Indexed</div>
        <div className="empty-text">No documents were processed in this analysis run.</div>
      </div>
    );
  }

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            <th>Document Name</th>
            <th>Total Pages</th>
            <th>Processed Coverage</th>
            <th>Facts Extracted</th>
            <th>Content Hash / ID</th>
            <th>Ingestion Time</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => {
            const factCount = factsPerDoc.get(doc.document_id) || 0;
            const docStatus = docStatusMap.get(doc.document_id);
            const processedPages = docStatus?.processed_pages?.length;
            const totalEligible = docStatus?.total_eligible_pages;

            return (
              <tr key={doc.document_id}>
                <td style={{ fontWeight: 600 }}>{doc.document_name}</td>
                <td style={{ fontFeatureSettings: "'tnum' 1" }}>{doc.total_pages} pages</td>
                <td>
                  {processedPages != null && totalEligible != null ? (
                    <span
                      className={`coverage-pill ${
                        processedPages === totalEligible ? 'complete' : 'partial'
                      }`}
                    >
                      {processedPages} / {totalEligible} eligible pages
                    </span>
                  ) : (
                    <span className="coverage-pill complete">{doc.total_pages} pages</span>
                  )}
                </td>
                <td>
                  <span className="tag-badge">{factCount} facts</span>
                </td>
                <td
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 11,
                    color: 'var(--text-muted)',
                  }}
                >
                  {doc.document_id}
                </td>
                <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                  {doc.created_at ? new Date(doc.created_at).toLocaleString() : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
