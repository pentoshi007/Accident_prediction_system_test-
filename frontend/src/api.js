import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export const fetchHealth = () => api.get('/health');
export const fetchClusters = () => api.get('/clusters');
export const fetchClusterDetail = (id) => api.get(`/clusters/${id}`);
export const predictSeverity = (data) => api.post('/predict', data);
export const fetchEda = (type) => api.get(`/eda/${type}`);
export const fetchModelMetrics = () => api.get('/model/metrics');
export const fetchEdaSummary = () => api.get('/eda/summary');
export const uploadCSV = (file) => {
  const fd = new FormData();
  fd.append('file', file);
  return api.post('/upload', fd);
};
export const runPipeline = () => api.post('/pipeline/run');

export default api;
