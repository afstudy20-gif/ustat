import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import ChartsPanel from './ChartsPanel'

afterEach(() => clearSession())

describe('ChartsPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<ChartsPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows chart type radio buttons and defaults to histogram', () => {
    installSession()
    render(<ChartsPanel />)
    expect(screen.getByRole('radio', { name: /histogram/i })).toBeChecked()
    expect(screen.getByRole('radio', { name: /scatter/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /boxplot/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /violin/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /^bar$/i })).toBeInTheDocument()
  })

  it('runs histogram generation and renders the plot on success', async () => {
    installSession()
    server.use(
      http.post('/api/charts/histogram', () =>
        HttpResponse.json({
          type: 'histogram',
          x: 'AGE',
          bins: [
            { x0: 40, x1: 50, count: 1 },
            { x0: 50, x1: 60, count: 2 },
          ],
          kde: [
            { x: 45, y: 0.01 },
            { x: 55, y: 0.02 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)

    const runButton = screen.getByRole('button', { name: /generate chart/i })
    await user.click(runButton)

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
    // Custom labels panel appears once plot data is present
    expect(screen.getByText('Custom Labels')).toBeInTheDocument()
  })

  it('runs a scatter chart with x/y selection', async () => {
    installSession()
    server.use(
      http.post('/api/charts/scatter', () =>
        HttpResponse.json({
          type: 'scatter',
          x: 'AGE',
          y: 'LDL',
          points: [
            { AGE: 55, LDL: 120 },
            { AGE: 62, LDL: 140 },
          ],
          regression: { line_x: [55, 62], line_y: [120, 140], r2: 0.8 },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)

    await user.click(screen.getByRole('radio', { name: /scatter/i }))
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.post('/api/charts/histogram', () =>
        HttpResponse.json({ detail: 'Column not found' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ChartsPanel />)
    await user.click(screen.getByRole('button', { name: /generate chart/i }))

    await waitFor(() => expect(screen.getByText('Column not found')).toBeInTheDocument())
  })
})
