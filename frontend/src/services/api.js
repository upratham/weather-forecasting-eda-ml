const BASE = '/api'

async function request(path, options = {}) {
  const r = await fetch(BASE + path, options)
  if (!r.ok) {
    let msg = `HTTP ${r.status}`
    try { msg = (await r.json()).detail || msg } catch (_) {}
    throw new Error(msg)
  }
  if (r.status === 204) return null
  return r.json()
}

export const geocode = (query) =>
  request(`/geocode?query=${encodeURIComponent(query)}`)

export const getWeather = ({ location, lat, lon }) => {
  const qs = location
    ? `location=${encodeURIComponent(location)}`
    : `lat=${lat}&lon=${lon}`
  return request(`/weather?${qs}`)
}

export const listQueries = () => request('/queries')

export const createQuery = (body) =>
  request('/queries', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

export const updateQuery = (id, body) =>
  request(`/queries/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

export const deleteQuery = (id) =>
  request(`/queries/${id}`, { method: 'DELETE' })

export const exportData = (fmt) => {
  window.open(`${BASE}/export?fmt=${fmt}`, '_blank')
}
