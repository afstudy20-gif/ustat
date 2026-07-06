import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import PowerPanel from './PowerPanel'

afterEach(() => clearSession())

describe('PowerPanel', () => {
  it('renders the empty state guide without any calculation run', () => {
    installSession()
    render(<PowerPanel />)
    expect(screen.getByText('Quick start guide')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Calculate Sample size/i })).toBeInTheDocument()
  })

  it('t-test, solve for n: runs and renders required sample size + power curve', async () => {
    installSession()
    server.use(
      http.post('/api/stats/power', () =>
        HttpResponse.json({
          result: 63.77,
          label: 'Required sample size per group',
          curve: [
            { n: 10, power: 0.2 },
            { n: 63, power: 0.8 },
            { n: 120, power: 0.95 },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<PowerPanel />)

    // Default test is t-test (2-grp), default solveFor is "n" -- just calculate.
    await user.click(screen.getByRole('button', { name: /Calculate Sample size/i }))

    await waitFor(() => expect(screen.getAllByText('64').length).toBeGreaterThan(0))
    expect(screen.getByText('Required sample size per group')).toBeInTheDocument()
    expect(screen.getByText('Power Curve')).toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })

  it('two proportions, solve for power: shows backend error message', async () => {
    installSession()
    server.use(
      http.post('/api/stats/power', () =>
        HttpResponse.json({ detail: 'Proportions must be between 0 and 1' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<PowerPanel />)

    await user.click(screen.getByRole('button', { name: 'Proportions' }))
    await user.click(screen.getByRole('button', { name: /Power \(1−β\)/i }))
    await user.click(screen.getByRole('button', { name: /Calculate Power/i }))

    await waitFor(() =>
      expect(screen.getByText('Proportions must be between 0 and 1')).toBeInTheDocument(),
    )
  })

  it('logistic regression, solve for effect size (OR): runs and renders the result', async () => {
    installSession()
    server.use(
      http.post('/api/stats/power', () =>
        HttpResponse.json({
          result: 1.85,
          label: 'Minimum detectable odds ratio',
          curve: [],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<PowerPanel />)

    await user.click(screen.getByRole('button', { name: 'Logistic' }))
    await user.click(screen.getByRole('button', { name: /Effect size/i }))
    await user.click(screen.getByRole('button', { name: /Calculate Effect size/i }))

    await waitFor(() => expect(screen.getAllByText('1.8500').length).toBeGreaterThan(0))
    expect(screen.getByText('Minimum detectable odds ratio')).toBeInTheDocument()
  })
})
