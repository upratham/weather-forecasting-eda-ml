import { useState } from 'react'
import styles from './SearchBar.module.css'

export default function SearchBar({ onSearch, onGPS, loading }) {
  const [value, setValue] = useState('')

  const submit = (e) => {
    e.preventDefault()
    const q = value.trim()
    if (q) onSearch(q)
  }

  return (
    <form className={styles.wrap} onSubmit={submit}>
      <div className={styles.inputRow}>
        <span className={styles.icon}>🔍</span>
        <input
          className={styles.input}
          type="text"
          placeholder="Search city, zip code, landmark, or coordinates..."
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={loading}
        />
        <button className={styles.searchBtn} type="submit" disabled={loading || !value.trim()}>
          {loading ? <span className={styles.spinner} /> : 'Search'}
        </button>
        <button
          className={styles.gpsBtn}
          type="button"
          title="Use my current location"
          onClick={onGPS}
          disabled={loading}
        >
          📍 My Location
        </button>
      </div>
    </form>
  )
}
