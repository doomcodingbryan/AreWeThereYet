import { useState, useEffect, useMemo, useRef } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { CountryResult } from './types'

const SUGGESTED_CATEGORIES = [
  { label: 'Tech Hubs', query: 'strong tech jobs, startup scene, and good public transit' },
  { label: 'Beach Living', query: 'warm beach lifestyle with safety and good healthcare' },
  { label: 'Family Friendly', query: 'safe family friendly country with quality schools and parks' },
  { label: 'Budget Friendly', query: 'affordable country with low cost of living and stable economy' },
  { label: 'Digital Nomad', query: 'digital nomad friendly with good internet and visa options' },
  { label: 'Nature + Outdoors', query: 'mountains hiking clean air and outdoor lifestyle' },
]

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [apiConnected, setApiConnected] = useState<boolean>(true)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [results, setResults] = useState<CountryResult[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [regionFilter, setRegionFilter] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'match' | 'cost' | 'safety'>('match')
  const [selected, setSelected] = useState<CountryResult | null>(null)
  const [explanations, setExplanations] = useState<Record<string, string>>({})
  const [explanationLoading, setExplanationLoading] = useState<boolean>(false)
  const [explanationRateLimited, setExplanationRateLimited] = useState<boolean>(false)
  const explainAbortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(data => {
        setUseLlm(data.use_llm)
        setApiConnected(true)
      })
      .catch(err => {
        console.error("Failed to fetch config, defaulting use_llm to false:", err)
        setApiConnected(false)
        setUseLlm(false)
      })
  }, [])

  const fetchExplanations = async (query: string, results: CountryResult[]): Promise<void> => {
    if (explainAbortRef.current) explainAbortRef.current.abort()
    const ctrl = new AbortController()
    explainAbortRef.current = ctrl
    setExplanations({})
    setExplanationRateLimited(false)
    setExplanationLoading(true)

    try {
      const response = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, countries: results }),
        signal: ctrl.signal,
      })
      if (!response.ok) return

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.country && data.explanation) {
                setExplanations(prev => ({ ...prev, [data.country]: data.explanation }))
              }
              if (data.error === 'rate_limited') {
                setExplanationRateLimited(true)
              }
            } catch { /* ignore malformed */ }
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') console.error(e)
    } finally {
      setExplanationLoading(false)
    }
  }

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)
    setError('')
    if (explainAbortRef.current) explainAbortRef.current.abort()

    if (value.trim() === '') {
      setResults([])
      setExplanations({})
      setExplanationLoading(false)
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: value })
      })
      const data = await response.json()
      if (!response.ok) {
        setApiConnected(false)
        setError(data.error || 'Search failed')
        setResults([])
        return
      }
      setApiConnected(true)
      const fetched = data.results || []
      setResults(fetched)
      if (useLlm && fetched.length) {
        fetchExplanations(value, fetched)
      }
    } catch {
      setApiConnected(false)
      setError('Search failed. Is the Flask backend running on port 5001?')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const regions = useMemo(() => {
    const vals = new Set<string>()
    results.forEach((r) => {
      if (r.metadata.region) vals.add(r.metadata.region)
    })
    return ['all', ...Array.from(vals).sort()]
  }, [results])

  const displayedResults = useMemo(() => {
    const filtered = results.filter((r) => (
      regionFilter === 'all' || r.metadata.region === regionFilter
    ))
    const numeric = (value: string): number => {
      const parsed = Number(value)
      return Number.isNaN(parsed) ? 0 : parsed
    }
    filtered.sort((a, b) => {
      if (sortBy === 'cost') {
        return numeric(a.metadata.cost_of_living_index) - numeric(b.metadata.cost_of_living_index)
      }
      if (sortBy === 'safety') {
        return numeric(b.metadata.safety_index) - numeric(a.metadata.safety_index)
      }
      return b.score - a.score
    })
    return filtered
  }, [results, regionFilter, sortBy])

  const getPercent = (value: string): number => {
    const n = Number(value)
    if (Number.isNaN(n)) return 0
    return Math.max(0, Math.min(100, n))
  }

  const hasMetricValue = (value: string): boolean => {
    if (value === undefined || value === null) return false
    return value.toString().trim() !== '' && !Number.isNaN(Number(value))
  }

  if (useLlm === null) return <></>

  return (
    <div className="full-body-container">

      {/* Search bar */}
      <div className="top-text">
        <div className="google-colors">
          <h1 id="google-4">Are</h1>
          <h1 id="google-3">We</h1>
          <h1 id="google-0-1">There</h1>
          <h1 id="google-0-2">Yet?</h1>
        </div>
        <h2 className="subheader">Find your true home</h2>

        <div
          className="input-box"
          onClick={() => document.getElementById('search-input')?.focus()}
        >
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Describe what you're looking for (e.g. warm beach, good transit)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch(searchTerm)}
          />
          <button
            type="button"
            className="search-submit-btn"
            onClick={() => handleSearch(searchTerm)}
          >Search</button>
        </div>

        <div className="suggested-categories">
          <p className="suggested-title">Suggested categories</p>
          <div className="category-chip-row">
            {SUGGESTED_CATEGORIES.map((item) => (
              <button
                key={item.label}
                type="button"
                className="category-chip"
                onClick={() => handleSearch(item.query)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {!apiConnected && (
        <div className="api-warning">
          Backend connection missing. Start it with <code>python src/app.py</code>, then refresh.
        </div>
      )}

      {/* Controls */}
      {!!results.length && (
        <div className="control-bar">
          <label>
            Region
            <select value={regionFilter} onChange={(e) => setRegionFilter(e.target.value)}>
              {regions.map((region) => (
                <option key={region} value={region}>
                  {region === 'all' ? 'All regions' : region}
                </option>
              ))}
            </select>
          </label>

          <label>
            Sort by
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value as 'match' | 'cost' | 'safety')}>
              <option value="match">Match</option>
              <option value="cost">Cost (low to high)</option>
              <option value="safety">Safety (high to low)</option>
            </select>
          </label>
        </div>
      )}

      {explanationRateLimited && (
        <p className="explanation-rate-limited-banner">
          AI insights unavailable, hourly token limit reached. Dimension tags still work. Try again later.
        </p>
      )}

      {loading && (
        <div className="search-loading" role="status" aria-live="polite">
          <span className="search-spinner" />
          <span>Refining your country matches...</span>
        </div>
      )}
      {!!error && <p className="search-error">{error}</p>}

      {/* Results */}
      <div id="answer-box">
        {displayedResults.map((res, idx) => {
          const m = res.metadata;
          return (
            <div
              key={res.country || idx}
              className="country-card"
              role="button"
              tabIndex={0}
              onClick={() => setSelected(res)}
              onKeyDown={(e) => e.key === 'Enter' && setSelected(res)}
            >
              <div className="card-header">
                <h3 className="country-title">{res.country}</h3>
                <span className="match-score">{Math.round(res.score * 100)}% Match</span>
              </div>

              <div className="metric-row">
                <div className="metric">
                  <span>Safety</span>
                  <div className="metric-track">
                    <div
                      className={`metric-fill ${hasMetricValue(m.safety_index) ? 'safety' : 'unknown'}`}
                      style={{ width: `${hasMetricValue(m.safety_index) ? getPercent(m.safety_index) : 100}%` }}
                    />
                  </div>
                  <small>{hasMetricValue(m.safety_index) ? m.safety_index : 'No data'}</small>
                </div>
                <div className="metric">
                  <span>Climate</span>
                  <div className="metric-track">
                    <div
                      className={`metric-fill ${hasMetricValue(m.climate_index) ? 'climate' : 'unknown'}`}
                      style={{ width: `${hasMetricValue(m.climate_index) ? getPercent(m.climate_index) : 100}%` }}
                    />
                  </div>
                  <small>{hasMetricValue(m.climate_index) ? m.climate_index : 'No data'}</small>
                </div>
                <div className="metric">
                  <span>Cost</span>
                  <div className="metric-track">
                    <div
                      className={`metric-fill ${hasMetricValue(m.cost_of_living_index) ? 'cost' : 'unknown'}`}
                      style={{ width: `${hasMetricValue(m.cost_of_living_index) ? getPercent(m.cost_of_living_index) : 100}%` }}
                    />
                  </div>
                  <small>{hasMetricValue(m.cost_of_living_index) ? m.cost_of_living_index : 'No data'}</small>
                </div>
              </div>

              <div className="country-metadata-grid">
                {m.region && (
                  <div className="meta-item">
                    <span className="meta-label">Region</span>
                    <span className="meta-value">{m.region}</span>
                  </div>
                )}
                {m.quality_of_life_index && (
                  <div className="meta-item">
                    <span className="meta-label">Quality of Life</span>
                    <span className="meta-value">{m.quality_of_life_index}</span>
                  </div>
                )}
                {m.cost_of_living_index && (
                  <div className="meta-item">
                    <span className="meta-label">Cost of Living</span>
                    <span className="meta-value">{m.cost_of_living_index}</span>
                  </div>
                )}
                {m.safety_index && (
                  <div className="meta-item">
                    <span className="meta-label">Safety</span>
                    <span className="meta-value">{m.safety_index}</span>
                  </div>
                )}
                {m.climate_index && (
                  <div className="meta-item">
                    <span className="meta-label">Climate</span>
                    <span className="meta-value">{m.climate_index}</span>
                  </div>
                )}
                {m.official_languages && (
                  <div className="meta-item">
                    <span className="meta-label">Language(s)</span>
                    <span className="meta-value">{m.official_languages}</span>
                  </div>
                )}
              </div>

              {res.dimensions && (res.dimensions.positive.length > 0 || res.dimensions.negative.length > 0) && (
                <div className="dimension-row">
                  {res.dimensions.positive.length > 0 && (
                    <div className="dimension-group">
                      <span className="dim-label">Matched on</span>
                      {res.dimensions.positive.map((d, i) => (
                        <span key={i} className="dim-tag dim-pos">
                          <span className="dim-num">Dim {d.dim}</span>
                          <span className="dim-terms">{d.terms.slice(0, 2).join(' · ')}</span>
                          <span className="dim-score">Score: +{d.contribution.toFixed(3)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                  {res.dimensions.negative.length > 0 && (
                    <div className="dimension-group">
                      <span className="dim-label">Contrasts</span>
                      {res.dimensions.negative.map((d, i) => (
                        <span key={i} className="dim-tag dim-neg">
                          <span className="dim-num">Dim {d.dim}</span>
                          <span className="dim-terms">{d.terms.slice(0, 2).join(' · ')}</span>
                          <span className="dim-score">Score: {d.contribution.toFixed(3)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {explanations[res.country] && (
                <div className="country-explanation">
                  <span className="explanation-label">AI Insight</span>
                  <p>{explanations[res.country]}</p>
                </div>
              )}
              {!explanations[res.country] && explanationLoading && (
                <div className="explanation-loading">
                  <span className="loading-dot" />
                  <span className="loading-dot" />
                  <span className="loading-dot" />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {selected && (
        <div className="country-modal-backdrop" onClick={() => setSelected(null)}>
          <div className="country-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{selected.country}</h3>
              <button type="button" onClick={() => setSelected(null)}>Close</button>
            </div>
            <p>
              <strong>Best for:</strong> {selected.metadata.region || 'Global fit'} lifestyle, with a{' '}
              {Math.round(selected.score * 100)}% match based on Reddit relocation discussions.
            </p>

            {selected.dimensions && (selected.dimensions.positive.length > 0 || selected.dimensions.negative.length > 0) && (
              <div className="modal-latent-section">
                <h4>SVD Dimension Activations</h4>
                <ul>
                  {selected.dimensions.positive.map((d) => (
                    <li key={d.dim}>Dim {d.dim} · {d.terms.slice(0, 3).join(', ')} — Score: +{d.contribution.toFixed(3)}</li>
                  ))}
                  {selected.dimensions.negative.map((d) => (
                    <li key={d.dim} style={{ color: '#8888aa' }}>Dim {d.dim} · {d.terms.slice(0, 3).join(', ')} — Score: {d.contribution.toFixed(3)}</li>
                  ))}
                </ul>
              </div>
            )}

            <ul className="modal-facts">
              <li>Quality of life: {selected.metadata.quality_of_life_index || 'N/A'}</li>
              <li>Cost of living: {selected.metadata.cost_of_living_index || 'N/A'}</li>
              <li>Safety: {selected.metadata.safety_index || 'N/A'}</li>
              <li>Healthcare: {selected.metadata.health_care_index || 'N/A'}</li>
              <li>Climate: {selected.metadata.climate_index || 'N/A'}</li>
              <li>Languages: {selected.metadata.official_languages || 'N/A'}</li>
              <li>Skilled-worker visa: {selected.metadata.skilled_worker_visa || 'N/A'}</li>
              <li>Visa program: {selected.metadata.visa_name || 'N/A'}</li>
            </ul>
          </div>
        </div>
      )}

    </div>
  )
}

export default App
