import { useState } from 'react'
import './DiagnoseForm.css'
export default function DiagnoseForm({ onDiagnose, onClear, loading }){
const [problem, setProblem] = useState('')
const [recipe,  setRecipe]  = useState('')

function handleSubmit() {
if (!problem.trim() || !recipe.trim()) return
onDiagnose({ problem, recipe, condition: 'full_system' })
}

function handleClear() {
setProblem('')
setRecipe('')
onClear()
}
return(
<div className="form-card">
<div className="field">
<label className="form-label">What went wrong?</label>
<textarea
placeholder="e.g. My hollandaise sauce is thin and 
won't thicken up..."
value={problem}
onChange={e => setProblem(e.target.value)}
rows={3}
/>
</div>
<div className="field">
<label className="form-label">Recipe name</label>
<input
type="text"
placeholder="e.g. Hollandaise Sauce"
value={recipe}
onChange={e => setRecipe(e.target.value)}
/>
</div>
<div className="btn-row">
<button
className="btn-primary"
onClick={handleSubmit}
disabled={loading || !problem.trim() || !recipe.trim()}>
{loading ? 'Analyzing...' : 'Diagnose'}
</button>
<button
className="btn-secondary"
onClick={handleClear}
disabled={loading}
>
Clear
</button>
</div>
</div>
)
}