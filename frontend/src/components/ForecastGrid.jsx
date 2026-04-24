import styles from './ForecastGrid.module.css'

const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']

export default function ForecastGrid({ forecast }) {
  if (!forecast?.length) return null
  return (
    <div className={styles.wrap}>
      <h3 className={styles.title}>7-Day Forecast</h3>
      <div className={styles.grid}>
        {forecast.map((day, i) => {
          const d = new Date(day.date + 'T12:00:00Z')
          const label = i === 0 ? 'Today' : DAYS[d.getUTCDay()]
          return (
            <div key={day.date} className={styles.card}>
              <p className={styles.day}>{label}</p>
              <p className={styles.date}>{day.date.slice(5)}</p>
              <span className={styles.emoji}>{day.emoji}</span>
              <p className={styles.desc}>{day.description}</p>
              <div className={styles.temps}>
                <span className={styles.hi}>{day.temp_max?.toFixed(0)}°</span>
                <span className={styles.lo}>{day.temp_min?.toFixed(0)}°</span>
              </div>
              {day.precipitation > 0 && (
                <p className={styles.precip}>🌧️ {day.precipitation} mm</p>
              )}
              <p className={styles.wind}>💨 {day.wind_max?.toFixed(0)} km/h</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
