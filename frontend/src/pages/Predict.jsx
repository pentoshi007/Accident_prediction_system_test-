import { useState } from 'react';
import { FiCpu, FiAlertTriangle } from 'react-icons/fi';
import RiskBadge from '../components/RiskBadge';
import { predictSeverity } from '../api';
import { mockPrediction } from '../mockData';

const WEATHER_OPTIONS = ['Clear', 'Rain', 'Fog', 'Snow', 'Wind', 'Other'];
const LIGHT_OPTIONS = ['Daylight', 'Darkness - lights lit', 'Darkness - lights unlit', 'Darkness - no lighting'];
const SURFACE_COND = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'];
const COLLISION_TYPES = ['Vehicle with vehicle collision', 'Collision with roadside objects', 'Collision with pedestrians', 'Rollover', 'Collision with animals', 'Other'];
const CAUSES = ['No distancing', 'Changing lane to the right', 'Changing lane to the left', 'Driving carelessly', 'No priority to vehicle', 'Moving Backward', 'Overtaking', 'Driving under the influence of drugs', 'Driving at high speed', 'Other'];
const EXPERIENCE = ['Below 1yr', '1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'unknown'];
const AGE_BANDS = ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown'];

const defaultForm = {
  cluster_id: 0, hour: 12, day_of_week: 3, is_night: 0,
  weather: 'Clear', num_vehicles: 2,
  light_conditions: 'Daylight', road_surface_conditions: 'Dry',
  type_of_collision: 'Vehicle with vehicle collision',
  cause_of_accident: 'No distancing',
  driving_experience: '5-10yr', age_band_of_driver: '18-30',
};

function SelectField({ label, name, value, options, onChange }) {
  return (
    <div>
      <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--clr-text-muted)' }}>{label}</label>
      <select name={name} value={value} onChange={onChange}
        className="w-full px-3 py-2 rounded-lg text-sm border outline-none focus:ring-2"
        style={{ background: 'var(--clr-surface-2)', borderColor: 'var(--clr-border)', color: 'var(--clr-text)', '--tw-ring-color': 'var(--clr-primary)' }}>
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

function NumberField({ label, name, value, min, max, onChange }) {
  return (
    <div>
      <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--clr-text-muted)' }}>{label}</label>
      <input type="number" name={name} value={value} min={min} max={max} onChange={onChange}
        className="w-full px-3 py-2 rounded-lg text-sm border outline-none focus:ring-2"
        style={{ background: 'var(--clr-surface-2)', borderColor: 'var(--clr-border)', color: 'var(--clr-text)', '--tw-ring-color': 'var(--clr-primary)' }} />
    </div>
  );
}

export default function Predict() {
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(mockPrediction);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [live, setLive] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null); setResult(null);
    try {
      const payload = { ...form, cluster_id: +form.cluster_id, hour: +form.hour, day_of_week: +form.day_of_week, is_night: +form.is_night, num_vehicles: +form.num_vehicles };
      const res = await predictSeverity(payload);
      setResult(res.data);
      setLive(true);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed');
    } finally { setLoading(false); }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-xl font-bold" style={{ color: 'var(--clr-text)' }}>Risk Prediction</h2>
        <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>
          Predict accident severity and ARI for specific conditions
          {!live && result && <span className="ml-2 text-xs px-2 py-0.5 rounded-full" style={{ background: '#f59e0b20', color: '#f59e0b' }}>Sample Result</span>}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Form */}
        <form onSubmit={handleSubmit} className="lg:col-span-3 rounded-xl border p-6 space-y-5"
          style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <NumberField label="Cluster ID" name="cluster_id" value={form.cluster_id} min={0} max={999} onChange={handleChange} />
            <NumberField label="Hour (0-23)" name="hour" value={form.hour} min={0} max={23} onChange={handleChange} />
            <NumberField label="Day of Week (0-6)" name="day_of_week" value={form.day_of_week} min={0} max={6} onChange={handleChange} />
            <SelectField label="Night?" name="is_night" value={form.is_night} options={[0, 1]} onChange={handleChange} />
            <NumberField label="Num Vehicles" name="num_vehicles" value={form.num_vehicles} min={1} max={10} onChange={handleChange} />
            <SelectField label="Weather" name="weather" value={form.weather} options={WEATHER_OPTIONS} onChange={handleChange} />
            <SelectField label="Light Conditions" name="light_conditions" value={form.light_conditions} options={LIGHT_OPTIONS} onChange={handleChange} />
            <SelectField label="Road Surface" name="road_surface_conditions" value={form.road_surface_conditions} options={SURFACE_COND} onChange={handleChange} />
            <SelectField label="Collision Type" name="type_of_collision" value={form.type_of_collision} options={COLLISION_TYPES} onChange={handleChange} />
            <SelectField label="Cause" name="cause_of_accident" value={form.cause_of_accident} options={CAUSES} onChange={handleChange} />
            <SelectField label="Driving Experience" name="driving_experience" value={form.driving_experience} options={EXPERIENCE} onChange={handleChange} />
            <SelectField label="Driver Age Band" name="age_band_of_driver" value={form.age_band_of_driver} options={AGE_BANDS} onChange={handleChange} />
          </div>

          <button type="submit" disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-lg text-sm font-semibold text-white transition-all"
            style={{ background: loading ? 'var(--clr-border)' : 'var(--clr-primary)' }}>
            <FiCpu size={16} />
            {loading ? 'Predicting...' : 'Run Prediction'}
          </button>
        </form>

        {/* Result */}
        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="rounded-xl border p-5 flex items-center gap-3"
              style={{ background: '#ef444415', borderColor: '#ef444440', color: '#ef4444' }}>
              <FiAlertTriangle size={18} />
              <p className="text-sm">{error}</p>
            </div>
          )}

          {result && (
            <>
              <div className="rounded-xl border p-6 text-center"
                style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
                <p className="text-xs uppercase tracking-wider mb-2" style={{ color: 'var(--clr-text-muted)' }}>
                  Predicted Severity
                </p>
                <p className="text-3xl font-bold mb-2" style={{ color: 'var(--clr-primary-light)' }}>
                  {result.predicted_label}
                </p>
                <RiskBadge tier={result.risk_tier} />
              </div>

              <div className="rounded-xl border p-6"
                style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
                <p className="text-xs uppercase tracking-wider mb-3" style={{ color: 'var(--clr-text-muted)' }}>
                  ARI Score
                </p>
                <div className="relative h-4 rounded-full overflow-hidden mb-2" style={{ background: 'var(--clr-surface-2)' }}>
                  <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                    style={{
                      width: `${(result.ari_score || 0) * 100}%`,
                      background: result.risk_tier === 'Critical' ? '#ef4444'
                        : result.risk_tier === 'Severe' ? '#f97316'
                        : result.risk_tier === 'Moderate' ? '#f59e0b' : '#22c55e',
                    }} />
                </div>
                <p className="text-2xl font-bold font-mono" style={{ color: 'var(--clr-text)' }}>
                  {result.ari_score}
                </p>
              </div>

              <div className="rounded-xl border p-6"
                style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
                <p className="text-xs uppercase tracking-wider mb-3" style={{ color: 'var(--clr-text-muted)' }}>
                  Severity Probabilities
                </p>
                <div className="space-y-2">
                  {Object.entries(result.severity_probabilities || {}).map(([label, prob]) => (
                    <div key={label} className="flex items-center gap-3">
                      <span className="text-xs w-24 shrink-0" style={{ color: 'var(--clr-text-muted)' }}>{label}</span>
                      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'var(--clr-surface-2)' }}>
                        <div className="h-full rounded-full" style={{ width: `${prob * 100}%`, background: 'var(--clr-primary)' }} />
                      </div>
                      <span className="text-xs font-mono w-12 text-right" style={{ color: 'var(--clr-text)' }}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border p-6"
                style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
                <p className="text-xs uppercase tracking-wider mb-3" style={{ color: 'var(--clr-text-muted)' }}>
                  ARI Weights
                </p>
                <div className="grid grid-cols-3 gap-3 text-center">
                  {Object.entries(result.weights || {}).map(([k, v]) => (
                    <div key={k}>
                      <p className="text-lg font-bold font-mono" style={{ color: 'var(--clr-primary-light)' }}>{v}</p>
                      <p className="text-xs" style={{ color: 'var(--clr-text-muted)' }}>{k.replace('_', ' ')}</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {!result && !error && (
            <div className="rounded-xl border p-10 text-center"
              style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
              <FiCpu size={40} className="mx-auto mb-3" style={{ color: 'var(--clr-border)' }} />
              <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>
                No prediction data available
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
