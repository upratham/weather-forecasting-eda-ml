import { useState } from 'react'
import { createQuery } from '../services/api'
import styles from './SaveQueryModal.module.css'

export default function SaveQueryModal({ weather, onClose, onSaved }) {
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [notes, setNotes] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const save = async () => {
    setError('')
    if (startDate && endDate && endDate < startDate) {
      setError('End date must be on or after start date.')
      return
    }
    setBusy(true)
    try {
      await createQuery({
        location_query: weather.location,
        location_name: weather.location,
        latitude: weather.latitude,
        longitude: weather.longitude,
        start_date: startDate || null,
        end_date: endDate || null,
        temperature_celsius: weather.current.temperature,
        weather_description: weather.current.description,
        humidity: weather.current.humidity,
        wind_speed: weather.current.wind_speed,
        notes: notes || null,
      })
      onSaved()
      onClose()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={styles.overlay} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className={styles.modal}>
        <h2 className={styles.title}>💾 Save to History</h2>
        <p className={styles.sub}>{weather.location} — {weather.current.temperature.toFixed(1)}°C {weather.current.emoji}</p>

        <label className={styles.label}>Date Range (optional — for historical reference)</label>
        <div className={styles.dateRow}>
          <input
            type="date"
            className={styles.input}
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            placeholder="Start date"
          />
          <span className={styles.dash}>–</span>
          <input
            type="date"
            className={styles.input}
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            placeholder="End date"
          />
        </div>

        <label className={styles.label}>Notes (optional)</label>
        <textarea
          className={styles.textarea}
          rows={3}
          placeholder="Add any notes about this location or weather..."
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />

        {error && <p className={styles.error}>{error}</p>}

        <div className={styles.buttons}>
          <button className={styles.cancelBtn} onClick={onClose}>Cancel</button>
          <button className={styles.saveBtn} onClick={save} disabled={busy}>
            {busy ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
