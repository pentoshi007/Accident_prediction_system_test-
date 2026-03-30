export default function StatCard({ icon, label, value, sub, color = 'var(--clr-primary)' }) {
  return (
    <div className="rounded-xl p-5 border transition-all duration-200 hover:translate-y-[-2px]"
      style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider mb-1"
            style={{ color: 'var(--clr-text-muted)' }}>{label}</p>
          <p className="text-2xl font-bold" style={{ color }}>{value}</p>
          {sub && <p className="text-xs mt-1" style={{ color: 'var(--clr-text-muted)' }}>{sub}</p>}
        </div>
        {icon && (
          <div className="w-10 h-10 rounded-lg flex items-center justify-center"
            style={{ background: `${color}20`, color }}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
