/**
 * Chat component — only rendered when USE_LLM = True in routes.py.
 *
 * RAG pipeline:
 *   1. User sends message
 *   2. Backend LLM modifies the query for IR → emits `search_term`
 *   3. Backend runs IR (search_engine) → emits `ir_results`
 *   4. Backend streams LLM answer grounded in retrieved docs → emits `content`
 *
 * `onSearchTerm` updates the search bar with the LLM-modified query.
 * `onResults` updates the country cards with the IR-retrieved results.
 */
import { useState, useRef, useEffect } from 'react'
import SearchIcon from './assets/mag.png'
import { CountryResult } from './types'

interface Message {
  text: string
  isUser: boolean
}

interface ChatProps {
  onSearchTerm: (term: string) => void
  onResults: (results: CountryResult[]) => void
}

function Chat({ onSearchTerm, onResults }: ChatProps): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages(prev => [...prev, { text, isUser: true }])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })

      if (!response.ok) {
        const data = await response.json()
        setMessages(prev => [...prev, { text: 'Error: ' + (data.error || response.status), isUser: false }])
        setLoading(false)
        return
      }

      let assistantText = ''
      setMessages(prev => [...prev, { text: '', isUser: false }])
      setLoading(false)

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
              // Step 2: LLM-modified query → update search bar
              if (data.search_term !== undefined) {
                onSearchTerm(data.search_term)
              }
              // Step 3: IR results → update country cards directly (no second API call)
              if (data.ir_results !== undefined) {
                onResults(data.ir_results)
              }
              if (data.error) {
                setMessages(prev => [...prev.slice(0, -1), { text: 'Error: ' + data.error, isUser: false }])
                return
              }
              // Step 4: streaming LLM answer
              if (data.content !== undefined) {
                assistantText += data.content
                setMessages(prev => [...prev.slice(0, -1), { text: assistantText, isUser: false }])
              }
            } catch { /* ignore malformed lines */ }
          }
        }
      }
    } catch {
      setMessages(prev => [...prev, { text: 'Something went wrong. Check the console.', isUser: false }])
      setLoading(false)
    }
  }

  return (
    <>
      <div id="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.isUser ? 'user' : 'assistant'}`}>
            <p>{msg.text}</p>
          </div>
        ))}
        {loading && (
          <div className="loading-indicator visible">
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-bar">
        <form className="input-row" onSubmit={sendMessage}>
          <img src={SearchIcon} alt="" />
          <input
            type="text"
            placeholder="Ask the AI about countries, visas, and relocation tradeoffs"
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={loading}
            autoComplete="off"
          />
          <button type="submit" disabled={loading}>Send</button>
        </form>
      </div>
    </>
  )
}

export default Chat
