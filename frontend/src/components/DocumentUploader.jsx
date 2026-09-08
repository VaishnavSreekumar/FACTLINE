import React, { useRef, useState } from 'react';

export default function DocumentUploader({ onAnalyze, isLoading }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFiles = (files) => {
    const pdfFiles = Array.from(files).filter(
      (file) => file.name.toLowerCase().endsWith('.pdf')
    );
    if (pdfFiles.length === 0) return;

    // Avoid duplicate files by name
    setSelectedFiles((prev) => {
      const existingNames = new Set(prev.map((f) => f.name));
      const newUnique = pdfFiles.filter((f) => !existingNames.has(f.name));
      return [...prev, ...newUnique];
    });
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const removeFile = (indexToRemove) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== indexToRemove));
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const handleSubmit = () => {
    if (selectedFiles.length > 0 && !isLoading) {
      onAnalyze(selectedFiles);
    }
  };

  return (
    <div className="upload-panel">
      <div className="upload-header">
        <h2>Document Ingestion</h2>
        <p>
          Upload PDF documents for cross-document fact extraction, dimensional normalization,
          and comparability reasoning.
        </p>
      </div>

      <div
        className={`dropzone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,application/pdf"
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="dropzone-text">Drop PDF files here or click to browse</div>
        <div className="dropzone-hint">
          Supports single or multi-document analysis (.pdf format)
        </div>
      </div>

      {selectedFiles.length > 0 && (
        <>
          <div className="file-list">
            {selectedFiles.map((file, idx) => (
              <div key={file.name + idx} className="file-item">
                <div className="file-info">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">({formatFileSize(file.size)})</span>
                </div>
                <button
                  className="btn-remove"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(idx);
                  }}
                  title="Remove file"
                  disabled={isLoading}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          <div className="upload-actions">
            <button
              className="btn"
              onClick={() => setSelectedFiles([])}
              disabled={isLoading}
            >
              Clear All
            </button>
            <button
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={isLoading}
            >
              {isLoading
                ? 'Analyzing Documents...'
                : `Analyze ${selectedFiles.length} Document${selectedFiles.length > 1 ? 's' : ''}`}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
