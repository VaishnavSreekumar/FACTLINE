import React, { useState } from 'react';
import Header from './components/Header';
import DocumentUploader from './components/DocumentUploader';
import AnalysisProgress from './components/AnalysisProgress';
import RelationshipsView from './components/RelationshipsView';
import FactsView from './components/FactsView';
import DocumentsView from './components/DocumentsView';

export default function App() {
  const [activeTab, setActiveTab] = useState('relationships');
  const [isLoading, setIsLoading] = useState(false);
  const [analyzingFileCount, setAnalyzingFileCount] = useState(0);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [error, setError] = useState(null);

  const loadAnalysisById = async (analysisId) => {
    setIsLoading(true);
    setError(null);
    try {
      const detailResp = await fetch(`/analysis/${analysisId}`);
      if (!detailResp.ok) {
        throw new Error(`Failed to retrieve analysis records for ID: ${analysisId}`);
      }
      const fullAnalysis = await detailResp.json();
      setCurrentAnalysis(fullAnalysis);
      setActiveTab('relationships');
    } catch (err) {
      console.error('Failed to load analysis:', err);
      setError(err.message || 'Could not load analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlId = params.get('analysis_id') || params.get('id');
    if (urlId) {
      loadAnalysisById(urlId);
    }
  }, []);

  const handleAnalyze = async (files) => {
    if (!files || files.length === 0) return;

    setIsLoading(true);
    setError(null);
    setAnalyzingFileCount(files.length);

    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file);
      });

      // 1. Submit POST /analysis
      const resp = await fetch('/analysis', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        let errMsg = `Analysis request failed (HTTP ${resp.status})`;
        try {
          const errorData = await resp.json();
          if (errorData.detail) {
            errMsg = errorData.detail;
          }
        } catch {
          // ignore
        }
        throw new Error(errMsg);
      }

      const summaryData = await resp.json();
      const analysisId = summaryData.analysis_id;

      if (!analysisId) {
        throw new Error('Analysis completed but did not return a valid analysis ID.');
      }

      // 2. Fetch full structured results via GET /analysis/{analysis_id}
      await loadAnalysisById(analysisId);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err.message || 'An unexpected error occurred while analyzing documents.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewAnalysis = () => {
    setCurrentAnalysis(null);
    setError(null);
  };

  const docCount = currentAnalysis?.documents?.length || 0;
  const factCount = currentAnalysis?.facts?.length || 0;
  const relCount = currentAnalysis?.relationships?.length || 0;

  // Extract page coverage & quota status if available
  const extractionStatus = currentAnalysis?.summary?.extraction_status;
  const totalPages =
    currentAnalysis?.documents?.reduce((acc, d) => acc + (d.total_pages || 0), 0) || 0;
  const extractionStatusType = extractionStatus?.status;
  const processedPages = extractionStatus?.total_processed_pages;
  const totalEligible = extractionStatus?.total_eligible_pages;

  const renderCoveragePill = () => {
    if (!extractionStatus) return null;

    if (extractionStatusType === 'COMPLETE') {
      return <span className="coverage-pill complete">Complete Analysis</span>;
    }
    if (extractionStatusType === 'PARTIAL_QUOTA') {
      return (
        <span className="coverage-pill partial">
          Partial Analysis · {processedPages ?? totalPages} / {totalEligible ?? totalPages} pages
        </span>
      );
    }
    if (extractionStatusType === 'QUOTA_EXHAUSTED') {
      return (
        <span className="coverage-pill partial">
          Quota Exhausted · {processedPages ?? 0} / {totalEligible ?? totalPages} pages
        </span>
      );
    }
    if (extractionStatusType === 'FAILED') {
      const pageInfo =
        processedPages != null && totalEligible != null
          ? ` · ${processedPages} / ${totalEligible} pages`
          : '';
      return (
        <span className="coverage-pill failed">
          Extraction Failed{pageInfo}
        </span>
      );
    }
    return (
      <span className="coverage-pill partial">
        Analysis · {processedPages ?? totalPages} / {totalEligible ?? totalPages} pages
      </span>
    );
  };

  return (
    <div className="app-container">
      <Header currentAnalysis={currentAnalysis} onNewAnalysis={handleNewAnalysis} />

      <main className="main-content">
        {error && (
          <div className="error-banner">
            <div>
              <strong>Analysis Notice:</strong> {error}
            </div>
            <button className="btn btn-sm" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}

        {isLoading ? (
          <AnalysisProgress fileCount={analyzingFileCount} />
        ) : !currentAnalysis ? (
          <DocumentUploader onAnalyze={handleAnalyze} isLoading={isLoading} />
        ) : (
          <div>
            {/* Analytical Summary Bar */}
            <div className="summary-bar">
              <div className="summary-metrics">
                <div className="metric-group">
                  <span className="metric-num">{docCount}</span>
                  <span className="metric-label">
                    {docCount === 1 ? 'Document' : 'Documents'}
                  </span>
                </div>

                <div className="metric-divider" />

                <div className="metric-group">
                  <span className="metric-num">{totalPages}</span>
                  <span className="metric-label">Total Pages</span>
                </div>

                {renderCoveragePill()}

                <div className="metric-divider" />

                <div className="metric-group">
                  <span className="metric-num">{factCount}</span>
                  <span className="metric-label">{factCount === 1 ? 'Fact' : 'Facts'}</span>
                </div>

                <div className="metric-divider" />

                <div className="metric-group">
                  <span className="metric-num">{relCount}</span>
                  <span className="metric-label">
                    {relCount === 1 ? 'Relationship' : 'Relationships'}
                  </span>
                </div>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div className="tabs-nav">
              <button
                className={`tab-btn ${activeTab === 'relationships' ? 'active' : ''}`}
                onClick={() => setActiveTab('relationships')}
              >
                Relationships
                <span className="tab-badge">{relCount}</span>
              </button>
              <button
                className={`tab-btn ${activeTab === 'facts' ? 'active' : ''}`}
                onClick={() => setActiveTab('facts')}
              >
                Facts
                <span className="tab-badge">{factCount}</span>
              </button>
              <button
                className={`tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
                onClick={() => setActiveTab('documents')}
              >
                Documents
                <span className="tab-badge">{docCount}</span>
              </button>
            </div>

            {/* Tab Views */}
            {activeTab === 'relationships' && (
              <RelationshipsView
                relationships={currentAnalysis.relationships || []}
                facts={currentAnalysis.facts || []}
                documents={currentAnalysis.documents || []}
              />
            )}

            {activeTab === 'facts' && (
              <FactsView
                facts={currentAnalysis.facts || []}
                documents={currentAnalysis.documents || []}
              />
            )}

            {activeTab === 'documents' && (
              <DocumentsView
                documents={currentAnalysis.documents || []}
                facts={currentAnalysis.facts || []}
                extractionStatus={extractionStatus}
              />
            )}
          </div>
        )}
      </main>
    </div>
  );
}
