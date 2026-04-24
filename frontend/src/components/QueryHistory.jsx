import { useState } from 'react'
import { updateQuery, deleteQuery } from '../services/api'
import styles from './QueryHistory.module.css'

export default function QueryHistory({ queries, onRefresh }) {
  const [editId, setEditId] = useState(null)
  const [editData, setEditData] = useState({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const startEdit = (q) => {
    setEditId(q.id)
    setEditData({ notes: q.notes || '', start_date: q.start_date || '', end_date: q.end_date || '' })
    setError('')
  }

  const saveEdit = async () => {
    setBusy(true)
    setError('')
    try {
      const payload = { notes: editData.notes || null }
      if (editData.start_date) payload.start_date = editData.start_date
      if (editData.end_date) payload.end_date = editData.end_date
      await updateQuery(editId, payload)
      setEditId(null)
      onRefresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const remove = async (id) => {
    if (!confirm('Delete this record?')) return
    setBusy(true)
    try {
      await deleteQuery(id)
      onRefresh()
    } catch (e) {
      alert(e.message)
    } finally {
      setBusy(false)
    }
  }

  if (!queries.length) {
    return (
      <div className={styles.empty}>
        <p>No saved queries yet. Search for a location and click <strong>Save to History</strong>.</p>
      </div>
    )
  }

  return (
    <div className={styles.wrap}>
      {error && <p className={styles.error}>{error}</p>}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Location</th>
              <th>Temp</th>
              <th>Condition</th>
              <th>Date Range</th>
              <th>Notes</th>
              <th>Saved</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {queries.map((q) =>
              editId === q.id ? (
                <tr key={q.id} className={styles.editRow}>
                  <td>{q.location_name}</td>
                  <td>{q.temperature_celsius?.toFixed(1)}°C</td>
                  <td>{q.weather_description}</td>
                  <td>
                    <input
                      type="date"
                      className={styles.dateInput}
                      value={editData.start_date}
                      onChange={(e) => setEditData({ ...editData, start_date: e.target.value })}
                    />
                    <span> – </span>
                    <input
                      type="date"
                      className={styles.dateInput}
                      value={editData.end_date}
                      onChange={(e) => setEditData({ ...editData, end_date: e.target.value })}
                    />
                  </td>
                  <td>
                    <textarea
                      className={styles.notesInput}
                      value={editData.notes}
                      onChange={(e) => setEditData({ ...editData, notes: e.target.value })}
                      rows={2}
                      placeholder="Add notes..."
                    />
                  </td>
                  <td>{new Date(q.created_at).toLocaleDateString()}</td>
                  <td className={styles.actions}>
                    <button className={styles.saveBtn} onClick={saveEdit} disabled={busy}>Save</button>
                    <button className={styles.cancelBtn} onClick={() => setEditId(null)}>Cancel</button>
                  </td>
                </tr>
              ) : (
                <tr key={q.id}>
                  <td>
                    <p className={styles.locName}>{q.location_name}</p>
                    <p className={styles.coords}>{q.latitude.toFixed(3)}°, {q.longitude.toFixed(3)}°</p>
                  </td>
                  <td>{q.temperature_celsius != null ? `${q.temperature_celsius.toFixed(1)}°C` : '—'}</td>
                  <td>{q.weather_description || '—'}</td>
                  <td>
                    {q.start_date && q.end_date
                      ? `${q.start_date} → ${q.end_date}`
                      : q.start_date || '—'}
                  </td>
                  <td className={styles.notes}>{q.notes || <span className={styles.noNotes}>—</span>}</td>
                  <td>{new Date(q.created_at).toLocaleDateString()}</td>
                  <td className={styles.actions}>
                    <button className={styles.editBtn} onClick={() => startEdit(q)}>✏️ Edit</button>
                    <button className={styles.deleteBtn} onClick={() => remove(q.id)} disabled={busy}>🗑️</button>
                  </td>
                </tr>
              )
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
