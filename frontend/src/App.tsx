import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import Chat from './Chat'
import { CountryResult } from './types'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [results, setResults] = useState<CountryResult[]>([])

  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(data => setUseLlm(data.use_llm))
      .catch(err => {
        console.error("Failed to fetch config, defaulting use_llm to false:", err)
        setUseLlm(false)
      })
  }, [])

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)

    if (value.trim() === '') {
      setResults([])
      return
    }

    const response = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: value })
    })
    const data = await response.json()
    setResults(data.results || [])
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
      </div>

      {/* Results */}
      <div id="answer-box">
        {results.map((res, idx) => {
          const m = res.metadata;
          return (
            <div key={res.country || idx} className="country-card">
              <div className="card-header">
                <h3 className="country-title">{res.country}</h3>
                <span className="match-score">{Math.round(res.score * 100)}% Match</span>
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

      {/* Chat (optional) */}
      {useLlm && <Chat onSearchTerm={handleSearch} />}
    </div>
  )
}

export default App