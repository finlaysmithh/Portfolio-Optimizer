import React from 'react'

const Section = ({title, children}) => (
  <section style={{padding: '2rem 0', borderBottom: '1px solid #eee'}}>
    <h2 style={{marginBottom: '0.5rem'}}>{title}</h2>
    <div>{children}</div>
  </section>
)

export default function App() {
  return (
    <div style={{maxWidth: 900, margin: '0 auto', padding: '2rem'}}>
      <header style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <h1>Portfolio Optimizer Pro</h1>
        <nav>
          <a href="#demo" style={{marginRight: 16}}>Live Demo</a>
          <a href="../docs/index.md">Docs</a>
        </nav>
      </header>
      <p style={{color:'#444'}}>A 5–8 stock optimizer targeting ≥3% alpha vs S&P 500 with factor constraints, regime overlay, liquidity guards, realistic costs, and backtesting.</p>

      <Section title="Case Study">
        <p>We use value, quality, momentum, size, and low-volatility factors with z-scoring and composite ranking, a simple EWMA covariance with shrinkage, and realistic constraints (sector/single-name caps, factor exposure minimums). Monthly rebalances include 5 bps per unit turnover costs.</p>
      </Section>

      <Section title="Live Demo" id="demo">
        <p>Open the Streamlit app locally:</p>
        <pre>make app</pre>
      </Section>

      <Section title="About/Contact">
        <p>See README for full setup. This site is a lightweight landing page built with Vite + React.</p>
      </Section>
    </div>
  )
}
