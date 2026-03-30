export default function Loader({ text = 'Loading...' }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-3">
      <div className="w-8 h-8 border-3 rounded-full animate-spin"
        style={{ borderColor: 'var(--clr-border)', borderTopColor: 'var(--clr-primary)' }} />
      <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>{text}</p>
    </div>
  );
}
