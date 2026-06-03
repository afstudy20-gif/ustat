import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import UpdatePrompt from './components/UpdatePrompt'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
    {/* Floating toast — visible on every screen (upload zone, power panel,
        main app) so users always see when a newer build is ready. */}
    <UpdatePrompt />
  </StrictMode>,
)
