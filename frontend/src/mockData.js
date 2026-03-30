// Mock/default data shown before the API responds

const TIERS = ['Low', 'Moderate', 'Severe', 'Critical'];
const WEATHERS = ['Clear', 'Rain', 'Fog', 'Wind', 'Other'];
const ENVS = ['Clear', 'Rain', 'Fog', 'Snow', 'Wind'];

function rand(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
function randFloat(min, max, dec = 3) { return +(Math.random() * (max - min) + min).toFixed(dec); }
function pick(arr) { return arr[rand(0, arr.length - 1)]; }

const INDIAN_CITIES = [
  { name: 'Delhi', lat: 28.6139, lon: 77.2090 },
  { name: 'Mumbai', lat: 19.0760, lon: 72.8777 },
  { name: 'Bangalore', lat: 12.9716, lon: 77.5946 },
  { name: 'Chennai', lat: 13.0827, lon: 80.2707 },
  { name: 'Hyderabad', lat: 17.3850, lon: 78.4867 },
  { name: 'Kolkata', lat: 22.5726, lon: 88.3639 },
  { name: 'Pune', lat: 18.5204, lon: 73.8567 },
  { name: 'Ahmedabad', lat: 23.0225, lon: 72.5714 },
  { name: 'Jaipur', lat: 26.9124, lon: 75.7873 },
  { name: 'Lucknow', lat: 26.8467, lon: 80.9462 },
  { name: 'Nagpur', lat: 21.1458, lon: 79.0882 },
  { name: 'Bhopal', lat: 23.2599, lon: 77.4126 },
  { name: 'Patna', lat: 25.6093, lon: 85.1376 },
  { name: 'Chandigarh', lat: 30.7333, lon: 76.7794 },
  { name: 'Coimbatore', lat: 11.0168, lon: 76.9558 },
];

function makeClusterFeature(id) {
  const city = INDIAN_CITIES[id % INDIAN_CITIES.length];
  const lat = city.lat + randFloat(-0.15, 0.15);
  const lon = city.lon + randFloat(-0.15, 0.15);
  const ari = randFloat(0.05, 0.95);
  let tier;
  if (ari < 0.3) tier = 'Low';
  else if (ari < 0.5) tier = 'Moderate';
  else if (ari < 0.7) tier = 'Severe';
  else tier = 'Critical';

  return {
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [lon, lat] },
    properties: {
      Cluster_ID: id,
      Incident_Count: rand(15, 320),
      ARI_Score: ari,
      Risk_Tier: tier,
      Pred_Severity: rand(1, 3),
      Env_Modifier: pick(ENVS),
    },
  };
}

export const mockClusters = {
  type: 'FeatureCollection',
  features: Array.from({ length: 22 }, (_, i) => makeClusterFeature(i)),
};

export const mockSummary = {
  total_records: 12847,
  date_range: '2020-01-01 to 2024-12-31',
  unique_areas: 48,
};

export const mockHealth = { status: 'ok', models_loaded: { random_forest: true, dbscan_labels: true, ari_data: true, processed_data: true } };

export const mockEda = {
  hourly: Object.fromEntries(Array.from({ length: 24 }, (_, h) => [h, rand(80, 600)])),
  weekly: { Monday: rand(300, 700), Tuesday: rand(300, 700), Wednesday: rand(300, 700), Thursday: rand(300, 700), Friday: rand(400, 800), Saturday: rand(500, 900), Sunday: rand(400, 800) },
  severity: { 'Slight Injury': rand(4000, 7000), 'Serious Injury': rand(2000, 4000), 'Fatal Injury': rand(500, 1500) },
  weather: { Clear: rand(3000, 6000), Rain: rand(1500, 3500), Fog: rand(800, 2000), Wind: rand(400, 1200), Other: rand(200, 800) },
  top_areas: Object.fromEntries(INDIAN_CITIES.slice(0, 10).map(c => [c.name, rand(200, 1200)])),
  collision_types: {
    'Vehicle with vehicle': rand(3000, 6000),
    'Collision with pedestrians': rand(1000, 3000),
    'Rollover': rand(500, 1500),
    'Roadside objects': rand(400, 1200),
    'With animals': rand(100, 500),
    'Other': rand(200, 800),
  },
  causes: {
    'No distancing': rand(2000, 4000),
    'Driving carelessly': rand(1500, 3500),
    'High speed': rand(1200, 3000),
    'No priority to vehicle': rand(800, 2000),
    'Overtaking': rand(600, 1500),
    'Changing lane': rand(400, 1200),
    'Moving backward': rand(200, 800),
    'Under influence': rand(300, 900),
    'Fatigue': rand(200, 700),
    'Other': rand(100, 500),
  },
  severity_by_weather: {
    Clear: { Slight: 3200, Serious: 1400, Fatal: 400 },
    Rain: { Slight: 1800, Serious: 1100, Fatal: 600 },
    Fog: { Slight: 600, Serious: 700, Fatal: 500 },
  },
};

export const mockMetrics = {
  accuracy: 0.847,
  classification_report: {
    'Slight Injury': { precision: 0.88, recall: 0.91, 'f1-score': 0.89 },
    'Serious Injury': { precision: 0.79, recall: 0.76, 'f1-score': 0.77 },
    'Fatal Injury': { precision: 0.82, recall: 0.78, 'f1-score': 0.80 },
  },
};

export const mockPrediction = {
  cluster_id: 3,
  predicted_severity: 2,
  predicted_label: 'Serious Injury',
  severity_probabilities: { 'Slight Injury': 0.2134, 'Serious Injury': 0.5821, 'Fatal Injury': 0.2045 },
  ari_score: 0.6237,
  risk_tier: 'Severe',
  weights: { W1_severity: 0.4521, W2_density: 0.2893, W3_environment: 0.2586 },
  input_conditions: { weather: 'Rain', hour: 17, env_score: 0.75 },
};
