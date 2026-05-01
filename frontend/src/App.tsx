import { useState, useEffect, useMemo, useRef } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { CountryResult } from './types'
import GlobeComponent from './GlobeComponent'
import { BackgroundGradientGlow } from './components/ui/background-gradient-glow'
import { Button } from './components/ui/button'
import { ArrowRight } from 'lucide-react'

interface LatentPoint {
  dim: number
  terms: string[]
  contribution: number
}

interface LatentDimensions {
  positive: LatentPoint[]
  negative: LatentPoint[]
}

function LatentDimensionChart({ dimensions }: { dimensions?: LatentDimensions }): JSX.Element | null {
  if (!dimensions) return null
  const positive = dimensions.positive || []
  const negative = dimensions.negative || []
  if (positive.length === 0 && negative.length === 0) return null

  const points: LatentPoint[] = [...positive, ...negative].sort(
    (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)
  )
  const maxAbs = Math.max(
    ...points.map((p) => Math.abs(p.contribution)),
    0.0001
  )

  return (
    <div className="latent-chart">
      <div className="latent-chart-header">
        <span className="latent-chart-title">Latent dimension contributions</span>
        <span className="latent-chart-legend">
          <span className="legend-dot legend-pos" /> aligns
          <span className="legend-dot legend-neg" /> contrasts
        </span>
      </div>
      <div className="latent-chart-rows">
        {points.map((p) => {
          const widthPct = (Math.abs(p.contribution) / maxAbs) * 48
          const barStyle =
            p.contribution >= 0
              ? { left: '50%', width: `${widthPct}%` }
              : { left: `${50 - widthPct}%`, width: `${widthPct}%` }
          const label = p.terms.slice(0, 3).join(' · ') || `Dim ${p.dim}`
          return (
            <div key={`lat-${p.dim}`} className="latent-chart-row">
              <div className="latent-chart-meta">
                <strong>{label}</strong>
                <small>Dim {p.dim}</small>
              </div>
              <div className="latent-chart-track">
                <div className="latent-chart-midline" />
                <div
                  className={`latent-chart-bar ${p.contribution >= 0 ? 'positive' : 'negative'}`}
                  style={barStyle}
                />
              </div>
              <span className={`latent-chart-value ${p.contribution >= 0 ? 'pos' : 'neg'}`}>
                {p.contribution >= 0 ? '+' : ''}
                {p.contribution.toFixed(3)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}


function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [apiConnected, setApiConnected] = useState<boolean>(true)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [searchedQuery, setSearchedQuery] = useState<string>('')
  const [interpretedQuery, setInterpretedQuery] = useState<string>('')
  const [results, setResults] = useState<CountryResult[]>([])
  const [useSvd, setUseSvd] = useState<boolean>(true)
  const [globeCollapsed, setGlobeCollapsed] = useState<boolean>(false)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [selected, setSelected] = useState<CountryResult | null>(null)
  const [synthesis, setSynthesis] = useState<string>('')
  const [synthesisLoading, setSynthesisLoading] = useState<boolean>(false)
  const [synthesisRateLimited, setSynthesisRateLimited] = useState<boolean>(false)
  const synthesisAbortRef = useRef<AbortController | null>(null)

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

  const fetchSynthesis = async (query: string, results: CountryResult[]): Promise<void> => {
    if (synthesisAbortRef.current) synthesisAbortRef.current.abort()
    const ctrl = new AbortController()
    synthesisAbortRef.current = ctrl
    setSynthesis('')
    setSynthesisRateLimited(false)
    setSynthesisLoading(true)

    try {
      const response = await fetch('/api/synthesize', {
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
              if (data.content) setSynthesis(prev => prev + data.content)
              if (data.error === 'rate_limited') setSynthesisRateLimited(true)
            } catch { /* ignore malformed */ }
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') console.error(e)
    } finally {
      setSynthesisLoading(false)
    }
  }

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)
    setError('')
    if (synthesisAbortRef.current) synthesisAbortRef.current.abort()

    if (value.trim() === '') {
      setResults([])
      setSearchedQuery('')
      setInterpretedQuery('')
      setSynthesis('')
      setSynthesisLoading(false)
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      setSearchedQuery(value.trim())
      setInterpretedQuery(value.trim())
      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: value, svd_weight: useSvd ? null : 0 })
      })
      const data = await response.json()
      if (!response.ok) {
        setApiConnected(false)
        setError(data.error || 'Search failed')
        setResults([])
        return
      }
      setApiConnected(true)
      const original = data.original_query || value.trim()
      const transformed = data.transformed_query || data.query || original
      setSearchedQuery(original)
      setInterpretedQuery(transformed)
      const fetched = data.results || []
      setResults(fetched)
      setGlobeCollapsed(true)
      if (useLlm && fetched.length) {
        fetchSynthesis(transformed, fetched)
      }
    } catch {
      setApiConnected(false)
      setError('Search failed. Is the Flask backend running on port 5001?')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const displayedResults = useMemo(() => {
    return [...results].sort((a, b) => b.score - a.score)
  }, [results])

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
    <BackgroundGradientGlow>
    <div className="full-body-container">

      {/* Hero — full viewport */}
      <div className="hero-section">
        <h1 className="hero-title">AreWeThereYet</h1>
        <Button
          className="group hero-cta-btn"
          onClick={() => document.getElementById('search-section')?.scrollIntoView({ behavior: 'smooth' })}
        >
          Ready to move?
          <ArrowRight
            className="-me-1 ms-2 opacity-60 transition-transform group-hover:translate-x-0.5"
            size={16}
            strokeWidth={2}
            aria-hidden="true"
          />
        </Button>
      </div>

      {/* Search — below the fold */}
      <div id="search-section" className="search-section">
        <div className="search-card">
          <p className="search-card-label">Imagine your perfect home:</p>
          <p className="search-card-hint">Describe what you're looking for in a country such as lifestyle, climate, cost of living, and more</p>
          <div
            className="input-box"
            onClick={() => document.getElementById('search-input')?.focus()}
          >
            <img src={SearchIcon} alt="search" />
            <input
              id="search-input"
              placeholder="e.g. warm weather, low cost of living, safe for families..."
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

          <div className="retrieval-filters">
            <span className="retrieval-label">Retrieval method:</span>
            <button
              type="button"
              className={`retrieval-btn${!useSvd ? ' active' : ''}`}
              onClick={() => setUseSvd(false)}
            >TF-IDF</button>
            <button
              type="button"
              className={`retrieval-btn${useSvd ? ' active' : ''}`}
              onClick={() => setUseSvd(true)}
            >TF-IDF + SVD</button>
          </div>
        </div>
      </div>

      {/* Large globe — slides up when results arrive */}
      {!!results.length && (
        <div className="globe-results-section">
          <p className="matches-to-label">matches to: <span className="matches-to-query">{searchedQuery}</span></p>
          <div className={`globe-card${globeCollapsed ? ' globe-card--collapsed' : ''}`}>
            <div className="globe-card-header">
              <span className="globe-card-title">Globe view</span>
              <button
                type="button"
                className="globe-toggle-btn"
                onClick={() => setGlobeCollapsed(c => !c)}
              >
                {globeCollapsed ? 'View globe' : 'Minimize'}
              </button>
            </div>
            <div className={`globe-body${globeCollapsed ? ' globe-body--collapsed' : ''}`}>
              <GlobeComponent results={results} size={Math.min(window.innerWidth * 0.75, 700)} />
            </div>
          </div>
        </div>
      )}

      {!apiConnected && (
        <div className="api-warning">
          Backend connection missing. Start it with <code>python src/app.py</code>, then refresh.
        </div>
      )}

      {!!searchedQuery && (
        <div className="ai-query-overview">
          <span className="overview-label">AI Overview</span>
          {synthesisRateLimited ? (
            <p className="synthesis-rate-limited">AI overview unavailable — hourly token limit reached.</p>
          ) : synthesis ? (
            <p className="synthesis-text">{synthesis}</p>
          ) : synthesisLoading ? (
            <p className="synthesis-loading">
              <span className="loading-dot" />
              <span className="loading-dot" />
              <span className="loading-dot" />
            </p>
          ) : (
            <p>You searched for: <strong>{searchedQuery}</strong></p>
          )}
          {!!interpretedQuery && interpretedQuery !== searchedQuery && (
            <p className="synthesis-query-hint">IR query: <em>{interpretedQuery}</em></p>
          )}
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

              <LatentDimensionChart dimensions={res.dimensions} />
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
    </BackgroundGradientGlow>
  )
}

export default App
