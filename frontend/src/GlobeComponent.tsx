import { useRef, useEffect, useMemo } from 'react'
import Globe from 'react-globe.gl'
import { CountryResult } from './types'
import { getCoords } from './countryCoords'

interface GlobePoint {
  lat: number
  lng: number
  country: string
  score: number
  result: CountryResult
}

interface Props {
  results: CountryResult[]
  size?: number
}

const METRIC_GRADIENTS: Record<string, string> = {
  Safety: 'linear-gradient(90deg, #43a047, #80e27e)',
  Climate: 'linear-gradient(90deg, #1e88e5, #64b5f6)',
  Cost: 'linear-gradient(90deg, #fb8c00, #ffcc80)',
}

function metricBar(label: string, value: string): string {
  const n = Number(value)
  const valid = !Number.isNaN(n) && value.trim() !== ''
  const pct = valid ? Math.max(0, Math.min(100, n)) : 0
  const fill = valid ? METRIC_GRADIENTS[label] : 'repeating-linear-gradient(45deg,#ccc,#ccc 4px,#e0e0e0 4px,#e0e0e0 8px)'
  return `
    <div style="flex:1;overflow:hidden">
      <div style="font-size:10px;color:#6a85a8;margin-bottom:3px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">${label}</div>
      <div style="background:#e8f0f7;border-radius:4px;height:5px;overflow:hidden">
        <div style="width:${pct}%;height:100%;background:${fill};border-radius:4px"></div>
      </div>
      <div style="font-size:10px;color:#2e3f5c;margin-top:3px">${valid ? value : 'No data'}</div>
    </div>
  `
}

function buildLabel(p: GlobePoint): string {
  const m = p.result.metadata
  const pct = Math.round(p.score * 100)
  const scoreColor = p.score > 0.7 ? '#00e676' : p.score > 0.4 ? '#ffca28' : '#ff7043'

  const metaItems = [
    m.region ? `<div style="display:flex;flex-direction:column;gap:1px"><span style="font-size:9px;color:#6a85a8;text-transform:uppercase;letter-spacing:0.07em;font-weight:700">Region</span><span style="font-size:11px;color:#1a253b;font-weight:500">${m.region}</span></div>` : '',
    m.quality_of_life_index ? `<div style="display:flex;flex-direction:column;gap:1px"><span style="font-size:9px;color:#6a85a8;text-transform:uppercase;letter-spacing:0.07em;font-weight:700">Quality of Life</span><span style="font-size:11px;color:#1a253b;font-weight:500">${m.quality_of_life_index}</span></div>` : '',
    m.official_languages ? `<div style="display:flex;flex-direction:column;gap:1px"><span style="font-size:9px;color:#6a85a8;text-transform:uppercase;letter-spacing:0.07em;font-weight:700">Languages</span><span style="font-size:11px;color:#1a253b;font-weight:500">${m.official_languages}</span></div>` : '',
    m.skilled_worker_visa ? `<div style="display:flex;flex-direction:column;gap:1px"><span style="font-size:9px;color:#6a85a8;text-transform:uppercase;letter-spacing:0.07em;font-weight:700">Skilled Visa</span><span style="font-size:11px;color:#1a253b;font-weight:500">${m.skilled_worker_visa}</span></div>` : '',
  ].filter(Boolean).join('')

  return `
    <div style="
      background:rgba(255,255,255,0.97);
      border:1.5px solid rgba(180,200,220,0.7);
      border-radius:16px;
      padding:14px 16px;
      width:260px;
      font-family:sans-serif;
      box-shadow:0 8px 28px rgba(80,130,180,0.18);
      pointer-events:none;
    ">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span style="font-size:15px;font-weight:700;color:#1a253b">${p.country}</span>
        <span style="font-size:12px;font-weight:700;color:${scoreColor};background:${scoreColor}18;border-radius:20px;padding:2px 10px">${pct}% Match</span>
      </div>

      <div style="display:flex;gap:10px;margin-bottom:12px">
        ${metricBar('Safety', m.safety_index)}
        ${metricBar('Climate', m.climate_index)}
        ${metricBar('Cost', m.cost_of_living_index)}
      </div>

      ${metaItems ? `<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;border-top:1px solid #e8f0f7;padding-top:10px">${metaItems}</div>` : ''}
    </div>
  `
}

export default function GlobeComponent({ results, size = 420 }: Props): JSX.Element {
  const globeRef = useRef<any>(null)

  const points: GlobePoint[] = useMemo(() => {
    return results
      .map(r => {
        const coords = getCoords(r.country)
        if (!coords) return null
        return { lat: coords[0], lng: coords[1], country: r.country, score: r.score, result: r }
      })
      .filter((p): p is GlobePoint => p !== null)
  }, [results])

  useEffect(() => {
    if (points.length > 0 && globeRef.current) {
      globeRef.current.pointOfView({ lat: points[0].lat, lng: points[0].lng, altitude: 2 }, 1000)
    }
  }, [points])

  return (
    <Globe
      ref={globeRef}
      width={size}
      height={size}
      globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
      backgroundColor="rgba(0,0,0,0)"
      atmosphereColor="#a8c8e8"
      atmosphereAltitude={0.15}
      animateIn={true}
      pointsData={points}
      pointLat="lat"
      pointLng="lng"
      pointColor={(d: object) => {
        const score = (d as GlobePoint).score
        if (score > 0.7) return '#00e676'
        if (score > 0.4) return '#ffca28'
        return '#ff7043'
      }}
      pointAltitude={(d: object) => (d as GlobePoint).score * 0.1}
      pointRadius={(d: object) => 0.4 + (d as GlobePoint).score * 0.6}
      pointLabel={(d: object) => buildLabel(d as GlobePoint)}
    />
  )
}
