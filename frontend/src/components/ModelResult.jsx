import './ModelResult.css'
const ITEMS = [
{ key: 'cause',
label: 'Cause',       
color: '#dc2626' },
{ key: 'solution', label: 'Solution',     color: '#059669' 

},
{ key: 'explanation', label: 'Explanation',  color: '#6366f1' },
]
export default function ModelResult({ modelName, data, labelStyle }) {
const parsed = data?.parsed_response || {}
return (
<div className="model-block">
<div className="model-label" style={labelStyle}>
{modelName}
</div>
{ITEMS.map(item => (
<div className="result-item" key={item.key}>
<div className="result-tag">
<span className="dot" style={{ background: item.color }} />
{item.label}
</div>
<div className="result-text">
{parsed[item.key] || 'No response available.'}
</div>
</div>
))}
</div>
)
}