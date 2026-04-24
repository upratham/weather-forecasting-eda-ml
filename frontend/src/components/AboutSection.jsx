import styles from './AboutSection.module.css'

export default function AboutSection() {
  return (
    <div className={styles.wrap}>
      <div className={styles.card}>
        <h3 className={styles.cardTitle}>👨‍💻 Developer</h3>
        <p className={styles.name}>Prathamesh Suhas Uravane</p>
        <p className={styles.role}>AI / ML Engineer</p>
        <div className={styles.stack}>
          {['Python', 'FastAPI', 'React', 'SQLAlchemy', 'scikit-learn', 'Open-Meteo API'].map((t) => (
            <span key={t} className={styles.tag}>{t}</span>
          ))}
        </div>
        <p className={styles.note}>
          This weather app was built as a Full Stack technical assessment, combining a FastAPI backend
          with a React frontend. It uses the Open-Meteo API for real-time weather data (no API key required),
          SQLite for persistent CRUD storage, and includes multi-format data export.
        </p>
      </div>

      <div className={styles.card}>
        <h3 className={styles.cardTitle}>🚀 About PM Accelerator</h3>
        <p className={styles.pmText}>
          <strong>Product Manager Accelerator (PMA)</strong> is a leading community-driven program designed to help
          professionals break into and advance within product management. Through mentorship from experienced PMs,
          structured coaching, hands-on projects, and a thriving peer community, PMA accelerates careers in
          product management across tech, fintech, healthtech, and beyond.
        </p>
        <p className={styles.pmText}>
          PMA offers bootcamps, 1-on-1 coaching, portfolio-building workshops, and job placement support —
          empowering aspiring and current PMs to land roles at top companies and build products that matter.
        </p>
        <a
          className={styles.linkedInBtn}
          href="https://www.linkedin.com/company/product-manager-accelerator"
          target="_blank"
          rel="noopener noreferrer"
        >
          🔗 View on LinkedIn
        </a>
      </div>

      <div className={styles.card}>
        <h3 className={styles.cardTitle}>⚡ Features</h3>
        <ul className={styles.featureList}>
          <li>🌍 Search by city, postal code, landmark, or GPS coordinates</li>
          <li>📍 Auto-detect current location via browser GPS</li>
          <li>🌡️ Real-time weather: temperature, humidity, wind, UV, pressure</li>
          <li>📅 7-day forecast with WMO weather codes</li>
          <li>💾 CRUD-persistent query history with date ranges and notes</li>
          <li>📤 Export to JSON, CSV, XML, and Markdown</li>
          <li>🗺️ Google Maps + YouTube integration per location</li>
          <li>📱 Fully responsive design (desktop, tablet, mobile)</li>
        </ul>
      </div>
    </div>
  )
}
