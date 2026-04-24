import { useState, useEffect, useCallback } from 'react'
import { getWeather, listQueries } from './services/api'
import SearchBar from './components/SearchBar'
import WeatherCard from './components/WeatherCard'
import ForecastGrid from './components/ForecastGrid'
import QueryHistory from './components/QueryHistory'
import ExportPanel from './components/ExportPanel'
import AboutSection from './components/AboutSection'
import SaveQueryModal from './components/SaveQueryModal'
import styles from './App.module.css'

const TABS = [
  { id: 'history', label: '📋 History' },
  { id: 'export',  label: '📤 Export' },
  { id: 'about',   label: 'ℹ️ About' },
]

export default function App() {
  const [weather, setWeather]       = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState('')
  const [activeTab, setActiveTab]   = useState('history')
  const [queries, setQueries]       = useState([])
  const [showSave, setShowSave]     = useState(false)

  const loadQueries = useCallback(async () => {
    try {
      const data = await listQueries()
      setQueries(data)
    } catch (_) {}
  }, [])

  useEffect(() => { loadQueries() }, [loadQueries])

  const fetchWeather = async (params) => {
    setError('')
    setLoading(true)
    try {
      const data = await getWeather(params)
      setWeather(data)
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e) {
      setError(e.message)
      setWeather(null)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = (location) => fetchWeather({ location })

  const handleGPS = () => {
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by your browser.')
      return
    }
    setLoading(true)
    navigator.geolocation.getCurrentPosition(
      ({ coords }) => fetchWeather({ lat: coords.latitude, lon: coords.longitude }),
      (err) => {
        setError(`Location access denied: ${err.message}`)
        setLoading(false)
      },
      { timeout: 10000 }
    )
  }

  return (
    <div className={styles.app}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>🌤️</span>
            <div>
              <h1 className={styles.logoTitle}>WeatherFlow</h1>
              <p className={styles.logoSub}>AI-Powered Weather Intelligence</p>
            </div>
          </div>
          <p className={styles.byLine}>by Prathamesh Uravane</p>
        </div>
      </header>

      <main className={styles.main}>
        {/* Search */}
        <section className={styles.searchSection}>
          <SearchBar onSearch={handleSearch} onGPS={handleGPS} loading={loading} />
        </section>

        {/* Error */}
        {error && (
          <div className={styles.errorBanner}>
            <span>⚠️</span>
            <span>{error}</span>
            <button className={styles.errorClose} onClick={() => setError('')}>✕</button>
          </div>
        )}

        {/* Loading skeleton */}
        {loading && !weather && (
          <div className={styles.skeletonWrap}>
            <div className={styles.skeleton} style={{ height: 240 }} />
            <div className={styles.skeleton} style={{ height: 120, marginTop: 16 }} />
          </div>
        )}

        {/* Weather results */}
        {weather && (
          <section className={styles.results}>
            <WeatherCard weather={weather} onSave={() => setShowSave(true)} />
            <ForecastGrid forecast={weather.forecast} />
          </section>
        )}

        {/* Save modal */}
        {showSave && weather && (
          <SaveQueryModal
            weather={weather}
            onClose={() => setShowSave(false)}
            onSaved={() => { loadQueries(); setActiveTab('history') }}
          />
        )}

        {/* Bottom tabs */}
        <section className={styles.bottomSection}>
          <div className={styles.tabs}>
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`${styles.tab} ${activeTab === t.id ? styles.tabActive : ''}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
                {t.id === 'history' && queries.length > 0 && (
                  <span className={styles.badge}>{queries.length}</span>
                )}
              </button>
            ))}
          </div>
          <div className={styles.tabContent}>
            {activeTab === 'history' && (
              <QueryHistory queries={queries} onRefresh={loadQueries} />
            )}
            {activeTab === 'export' && <ExportPanel />}
            {activeTab === 'about'   && <AboutSection />}
          </div>
        </section>
      </main>

      <footer className={styles.footer}>
        <p>WeatherFlow · Built by Prathamesh Suhas Uravane · Powered by <a href="https://open-meteo.com" target="_blank" rel="noopener noreferrer">Open-Meteo</a> (free, no API key required)</p>
      </footer>
    </div>
  )
}
