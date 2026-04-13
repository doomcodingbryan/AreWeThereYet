import { useState, useEffect, useMemo } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import Chat from './Chat'
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

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)
    setError('')

    if (value.trim() === '') {
      setResults([])
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
      setResults(data.results || [])
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
    <div className={`full-body-container ${useLlm ? 'llm-mode' : ''}`}>

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
            onChange={(e) => handleSearch(e.target.value)}
          />
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

      {/* Chat (optional) */}
      {useLlm && <Chat onSearchTerm={handleSearch} />}
    </div>
  )
}

export default App