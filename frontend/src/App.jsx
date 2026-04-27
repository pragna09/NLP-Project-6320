import { useState }     from 'react'
import DiagnoseForm     from './components/DiagnoseForm'
import ModelResult      from './components/ModelResult'
import './App.css' 


export default function App() {
const [results, setResults] = useState(null)
const [loading, setLoading] = useState(false)
const [error,   setError]   = useState(null)
async function handleDiagnose(formData) {
setLoading(true)
setError(null)
setResults(null)
try {
const res = await fetch('http://localhost:8000/diagnose', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify(formData)
})
if (!res.ok) throw new Error(`Server error: ${res.status}`)
const data = await res.json()
setResults(data)
} catch (err) {
setError(err.message)
} finally {
setLoading(false)
}
}
function handleClear() {
setResults(null)
setError(null)
}
return (
<div className="app">
<div className="header">

  <span className="logo">R.E.C.I.P.E.</span>
<span className="badge">Kitchen Crisis Resolver</span>
<p className="subtitle">Describe your problem and get 
a diagnosis</p>
</div>
<div className="page">
<DiagnoseForm
onDiagnose={handleDiagnose}
onClear={handleClear}
loading={loading}
/>
{error && (
<div className="error-box">Error: {error}</div>
)}
{loading && (
<div className="loading">
<div className="spinner" />
Analyzing with all 3 models...
</div>
)}
{results && (
<div>
<p className="results-title">Diagnosis</p>
<ModelResult
modelName="LLaMA 3.3 70B"
data={results.llama_big}
labelStyle={{
background: 'linear-gradient(135deg, #4338ca, #6d28d9)',
color: '#fff'
}}
/>
<div className="divider" />
<ModelResult
modelName="LLaMA 3.1 8B"
data={results.llama_fast}
labelStyle={{
background: '#ede9fe',
color: '#4338ca',
border: '1px solid #c4b5fd'
}}
/>
<div className="divider" />
<ModelResult
modelName="GPT-OSS 20B"
data={results.gpt_oss}
labelStyle={{
background: '#f0fdf4',
color: '#166534',
border: '1px solid #86efac'
}}
/>
</div>
)}
</div>
</div>
)
}