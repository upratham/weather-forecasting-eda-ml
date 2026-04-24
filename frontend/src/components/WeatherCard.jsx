import styles from './WeatherCard.module.css'

const DIR = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
const windDir = (deg) => DIR[Math.round(deg / 22.5) % 16]

export default function WeatherCard({ weather, onSave }) {
  const { location, latitude, longitude, current } = weather
  const c = current
  const tempF = (c.temperature * 9 / 5 + 32).toFixed(1)

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div>
          <p className={styles.location}>{location}</p>
          <p className={styles.coords}>{latitude.toFixed(4)}°, {longitude.toFixed(4)}°</p>
          <p className={styles.time}>Last updated: {c.time?.replace('T', ' ')}</p>
        </div>
        <div className={styles.emojiWrap}>
          <span className={styles.emoji}>{c.emoji}</span>
          <p className={styles.description}>{c.description}</p>
        </div>
      </div>

      <div className={styles.tempRow}>
        <span className={styles.tempC}>{c.temperature.toFixed(1)}°C</span>
        <span className={styles.tempF}>{tempF}°F</span>
        <span className={styles.feelsLike}>Feels like {c.feels_like.toFixed(1)}°C</span>
      </div>

      <div className={styles.grid}>
        <Stat icon="💧" label="Humidity" value={`${c.humidity}%`} />
        <Stat icon="💨" label="Wind" value={`${c.wind_speed} km/h ${windDir(c.wind_direction)}`} />
        <Stat icon="🌡️" label="Pressure" value={`${c.pressure?.toFixed(0)} hPa`} />
        <Stat icon="🌧️" label="Precipitation" value={`${c.precipitation} mm`} />
        <Stat icon="☁️" label="Cloud Cover" value={`${c.cloud_cover}%`} />
        <Stat icon="🔆" label="UV Index" value={c.uv_index} />
      </div>

      <div className={styles.actions}>
        <a
          className={styles.mapLink}
          href={`https://www.google.com/maps?q=${latitude},${longitude}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          🗺️ View on Google Maps
        </a>
        <a
          className={styles.mapLink}
          href={`https://www.youtube.com/results?search_query=${encodeURIComponent(location + ' travel weather')}`}
          target="_blank"
          rel="noopener noreferrer"
        >
          ▶️ YouTube Videos
        </a>
        <button className={styles.saveBtn} onClick={onSave}>
          💾 Save to History
        </button>
      </div>
    </div>
  )
}

function Stat({ icon, label, value }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statIcon}>{icon}</span>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statValue}>{value ?? '—'}</span>
    </div>
  )
}
