import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import Chat from './Chat'

interface Post {
  id: string
  title: string
  body: string
  score: number
  subreddit: string
  num_comments: number
  countries: string[]
}

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [posts, setPosts] = useState<Post[]>([])

  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(data => setUseLlm(data.use_llm))
  }, [])

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)

    if (value.trim() === '') {
      setPosts([])
      return
    }

    const response = await fetch(`/api/posts?q=${encodeURIComponent(value)}`)
    const data: Post[] = await response.json()
    setPosts(data)
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

        <div
          className="input-box"
          onClick={() => document.getElementById('search-input')?.focus()}
        >
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Search Reddit posts..."
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Results */}
      <div id="answer-box">
        {posts.map((post) => (
          <div key={post.id} className="episode-item">
            <h3 className="episode-title">{post.title}</h3>

            <p className="episode-desc">
              {post.body ? post.body.slice(0, 200) + '...' : 'No content'}
            </p>

            <p className="episode-rating">
              Score: {post.score} | Comments: {post.num_comments}
            </p>

            <p>
              Countries:{' '}
              {post.countries.length > 0
                ? post.countries.join(', ')
                : 'None'}
            </p>

            <p style={{ fontSize: '0.8em', opacity: 0.7 }}>
              r/{post.subreddit}
            </p>
          </div>
        ))}
      </div>

      {/* Chat (optional) */}
      {useLlm && <Chat onSearchTerm={handleSearch} />}
    </div>
  )
}

export default App