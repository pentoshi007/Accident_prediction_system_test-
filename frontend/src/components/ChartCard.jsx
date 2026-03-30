export default function ChartCard({ title, children, className = '' }) {
  return (
    <div className={`rounded-xl border p-5 ${className}`}
      style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
      {title && (
        <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--clr-text)' }}>{title}</h3>
      )}
      {children}
    </div>
  );
}
