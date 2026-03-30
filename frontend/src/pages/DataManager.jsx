import { useState, useRef } from 'react';
import { FiUploadCloud, FiPlay, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { uploadCSV, runPipeline } from '../api';

export default function DataManager() {
  const fileRef = useRef();
  const [uploadStatus, setUploadStatus] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setUploading(true); setUploadStatus(null);
    try {
      const res = await uploadCSV(file);
      setUploadStatus({ ok: true, msg: `Uploaded: ${res.data.message} (${(res.data.size_bytes / 1024).toFixed(1)} KB)` });
    } catch (err) {
      setUploadStatus({ ok: false, msg: err.response?.data?.error || 'Upload failed' });
    } finally { setUploading(false); }
  };

  const handlePipeline = async () => {
    setRunning(true); setPipelineStatus(null);
    try {
      const res = await runPipeline();
      setPipelineStatus({ ok: true, msg: res.data.message });
    } catch (err) {
      setPipelineStatus({ ok: false, msg: err.response?.data?.error || 'Pipeline failed' });
    } finally { setRunning(false); }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h2 className="text-xl font-bold" style={{ color: 'var(--clr-text)' }}>Data Manager</h2>
        <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>Upload datasets and run the ML pipeline</p>
      </div>

      {/* Upload */}
      <div className="rounded-xl border p-6 space-y-4"
        style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--clr-primary)20', color: 'var(--clr-primary)' }}>
            <FiUploadCloud size={20} />
          </div>
          <div>
            <h3 className="text-sm font-semibold" style={{ color: 'var(--clr-text)' }}>Upload CSV Dataset</h3>
            <p className="text-xs" style={{ color: 'var(--clr-text-muted)' }}>Upload a road accident CSV file for processing</p>
          </div>
        </div>

        <div className="flex gap-3">
          <input ref={fileRef} type="file" accept=".csv"
            className="flex-1 text-sm file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:cursor-pointer"
            style={{ color: 'var(--clr-text-muted)' }} />
          <button onClick={handleUpload} disabled={uploading}
            className="px-5 py-2 rounded-lg text-sm font-medium text-white shrink-0"
            style={{ background: uploading ? 'var(--clr-border)' : 'var(--clr-primary)' }}>
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>

        {uploadStatus && (
          <div className="flex items-center gap-2 text-sm" style={{ color: uploadStatus.ok ? '#22c55e' : '#ef4444' }}>
            {uploadStatus.ok ? <FiCheckCircle /> : <FiXCircle />}
            {uploadStatus.msg}
          </div>
        )}
      </div>

      {/* Pipeline */}
      <div className="rounded-xl border p-6 space-y-4"
        style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center"
            style={{ background: '#22c55e20', color: '#22c55e' }}>
            <FiPlay size={20} />
          </div>
          <div>
            <h3 className="text-sm font-semibold" style={{ color: 'var(--clr-text)' }}>Run ML Pipeline</h3>
            <p className="text-xs" style={{ color: 'var(--clr-text-muted)' }}>
              Preprocess → EDA → DBSCAN → Random Forest → ARI
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {['Preprocess', 'EDA', 'DBSCAN', 'Classifier', 'ARI'].map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <span className="px-3 py-1.5 rounded-lg text-xs font-medium"
                style={{ background: 'var(--clr-surface-2)', color: 'var(--clr-text-muted)' }}>
                {step}
              </span>
              {i < 4 && <span style={{ color: 'var(--clr-border)' }}>→</span>}
            </div>
          ))}
        </div>

        <button onClick={handlePipeline} disabled={running}
          className="w-full py-3 rounded-lg text-sm font-semibold text-white flex items-center justify-center gap-2"
          style={{ background: running ? 'var(--clr-border)' : '#22c55e' }}>
          <FiPlay size={16} />
          {running ? 'Running pipeline... (this may take a few minutes)' : 'Run Full Pipeline'}
        </button>

        {pipelineStatus && (
          <div className="flex items-center gap-2 text-sm" style={{ color: pipelineStatus.ok ? '#22c55e' : '#ef4444' }}>
            {pipelineStatus.ok ? <FiCheckCircle /> : <FiXCircle />}
            {pipelineStatus.msg}
          </div>
        )}
      </div>
    </div>
  );
}
