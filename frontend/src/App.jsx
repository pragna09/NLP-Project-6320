// import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from './assets/vite.svg'
// import heroImg from './assets/hero.png'
// import './App.css'

// function App() {
//   const [count, setCount] = useState(0)

//   return (
//     <>
//       <section id="center">
//         <div className="hero">
//           <img src={heroImg} className="base" width="170" height="179" alt="" />
//           <img src={reactLogo} className="framework" alt="React logo" />
//           <img src={viteLogo} className="vite" alt="Vite logo" />
//         </div>
//         <div>
//           <h1>Get started</h1>
//           <p>
//             Edit <code>src/App.jsx</code> and save to test <code>HMR</code>
//           </p>
//         </div>
//         <button
//           className="counter"
//           onClick={() => setCount((count) => count + 1)}
//         >
//           Count is {count}
//         </button>
//       </section>

//       <div className="ticks"></div>

//       <section id="next-steps">
//         <div id="docs">
//           <svg className="icon" role="presentation" aria-hidden="true">
//             <use href="/icons.svg#documentation-icon"></use>
//           </svg>
//           <h2>Documentation</h2>
//           <p>Your questions, answered</p>
//           <ul>
//             <li>
//               <a href="https://vite.dev/" target="_blank">
//                 <img className="logo" src={viteLogo} alt="" />
//                 Explore Vite
//               </a>
//             </li>
//             <li>
//               <a href="https://react.dev/" target="_blank">
//                 <img className="button-icon" src={reactLogo} alt="" />
//                 Learn more
//               </a>
//             </li>
//           </ul>
//         </div>
//         <div id="social">
//           <svg className="icon" role="presentation" aria-hidden="true">
//             <use href="/icons.svg#social-icon"></use>
//           </svg>
//           <h2>Connect with us</h2>
//           <p>Join the Vite community</p>
//           <ul>
//             <li>
//               <a href="https://github.com/vitejs/vite" target="_blank">
//                 <svg
//                   className="button-icon"
//                   role="presentation"
//                   aria-hidden="true"
//                 >
//                   <use href="/icons.svg#github-icon"></use>
//                 </svg>
//                 GitHub
//               </a>
//             </li>
//             <li>
//               <a href="https://chat.vite.dev/" target="_blank">
//                 <svg
//                   className="button-icon"
//                   role="presentation"
//                   aria-hidden="true"
//                 >
//                   <use href="/icons.svg#discord-icon"></use>
//                 </svg>
//                 Discord
//               </a>
//             </li>
//             <li>
//               <a href="https://x.com/vite_js" target="_blank">
//                 <svg
//                   className="button-icon"
//                   role="presentation"
//                   aria-hidden="true"
//                 >
//                   <use href="/icons.svg#x-icon"></use>
//                 </svg>
//                 X.com
//               </a>
//             </li>
//             <li>
//               <a href="https://bsky.app/profile/vite.dev" target="_blank">
//                 <svg
//                   className="button-icon"
//                   role="presentation"
//                   aria-hidden="true"
//                 >
//                   <use href="/icons.svg#bluesky-icon"></use>
//                 </svg>
//                 Bluesky
//               </a>
//             </li>
//           </ul>
//         </div>
//       </section>

//       <div className="ticks"></div>
//       <section id="spacer"></section>
//     </>
//   )
// }

// export default App


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