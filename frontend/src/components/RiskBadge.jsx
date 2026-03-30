const tierColors = {
  Low: { bg: '#22c55e20', text: '#22c55e' },
  Moderate: { bg: '#f59e0b20', text: '#f59e0b' },
  Severe: { bg: '#f9731620', text: '#f97316' },
  Critical: { bg: '#ef444420', text: '#ef4444' },
};

export default function RiskBadge({ tier }) {
  const c = tierColors[tier] || tierColors.Low;
  return (
    <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold"
      style={{ background: c.bg, color: c.text }}>
      {tier}
    </span>
  );
}
