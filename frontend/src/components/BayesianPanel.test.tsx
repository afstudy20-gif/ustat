import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import BayesianPanel from './BayesianPanel'

afterEach(() => clearSession())

describe('BayesianPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<BayesianPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('defaults to one-sample t-test and shows the mu input', () => {
    installSession()
    render(<BayesianPanel />)
    expect(screen.getByDisplayValue('Bayesian One-Sample t-test')).toBeInTheDocument()
    expect(screen.getByText('Test Value (mu)')).toBeInTheDocument()
  })

  it('runs one-sample Bayesian t-test and renders BF10/BF01', async () => {
    installSession()
    server.use(
      http.post('/api/bayesian', () =>
        HttpResponse.json({
          analysis: 'Bayesian One-Sample t-test',
          n: 3,
          bf10: 12.5,
          bf01: 0.08,
          interpretation: 'Strong evidence for H1',
          statistic_label: 't',
          statistic_value: 3.1,
          df: 2,
          effect_size_label: 'd',
          effect_size_value: 1.4,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.click(screen.getByRole('button', { name: /compute bayes factor/i }))

    await waitFor(() => expect(screen.getByText('12.5000')).toBeInTheDocument())
    expect(screen.getByText('0.0800')).toBeInTheDocument()
    expect(screen.getByText(/Strong evidence for H1/)).toBeInTheDocument()
  })

  it('shows the grouping-variable selector for independent t-test', async () => {
    installSession()
    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.selectOptions(screen.getByDisplayValue('Bayesian One-Sample t-test'), 'ttest_ind')
    expect(screen.getByText('Grouping Variable')).toBeInTheDocument()
  })

  it('disables Compute for regression until predictors are selected', async () => {
    installSession()
    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.selectOptions(screen.getByDisplayValue('Bayesian One-Sample t-test'), 'regression')
    expect(screen.getByRole('button', { name: /compute bayes factor/i })).toBeDisabled()

    const multi = screen.getByText('Predictors (numeric)').closest('div')!.querySelector('select')!
    await user.selectOptions(multi, ['LDL'])
    expect(screen.getByRole('button', { name: /compute bayes factor/i })).toBeEnabled()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/bayesian', () =>
        HttpResponse.json({ detail: 'Invalid outcome column' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<BayesianPanel />)
    await user.click(screen.getByRole('button', { name: /compute bayes factor/i }))

    await waitFor(() => expect(screen.getByText('Invalid outcome column')).toBeInTheDocument())
  })
})
