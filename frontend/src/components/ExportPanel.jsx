import { exportData } from '../services/api'
import styles from './ExportPanel.module.css'

const FORMATS = [
  { fmt: 'json',     label: 'JSON',     icon: '{}',   desc: 'Machine-readable structured data' },
  { fmt: 'csv',      label: 'CSV',      icon: '📊',   desc: 'Spreadsheet-compatible format' },
  { fmt: 'xml',      label: 'XML',      icon: '🏷️',  desc: 'Extensible markup format' },
  { fmt: 'markdown', label: 'Markdown', icon: '📝',   desc: 'Formatted text table' },
]

export default function ExportPanel() {
  return (
    <div className={styles.wrap}>
      <h3 className={styles.title}>Export Your Weather Data</h3>
      <p className={styles.sub}>Download all saved queries in your preferred format.</p>
      <div className={styles.grid}>
        {FORMATS.map(({ fmt, label, icon, desc }) => (
          <button key={fmt} className={styles.card} onClick={() => exportData(fmt)}>
            <span className={styles.icon}>{icon}</span>
            <span className={styles.label}>{label}</span>
            <span className={styles.desc}>{desc}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
