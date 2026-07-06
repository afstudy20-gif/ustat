import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import CategoricalTestsPanel from './CategoricalTestsPanel'

afterEach(() => clearSession())

describe('CategoricalTestsPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<CategoricalTestsPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('defaults to the binomial test with a null-proportion input', () => {
    installSession()
    render(<CategoricalTestsPanel />)
    expect(screen.getByRole('radio', { name: 'Binomial test' })).toBeChecked()
    expect(screen.getByText('Expected proportion')).toBeInTheDocument()
  })

  it('switches to two-proportions and shows the group column selector', async () => {
    installSession()
    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('radio', { name: 'Two proportions z-test' }))
    expect(screen.getByText('Group column')).toBeInTheDocument()
  })

  it('runs the binomial test and renders the result card', async () => {
    installSession()
    server.use(
      http.post('/api/categorical/binomial', () =>
        HttpResponse.json({
          test: 'Binomial test',
          interpretation: 'Observed proportion differs from expected.',
          significant: true,
          successes: 2,
          n: 3,
          observed_p: 0.667,
          expected_p: 0.5,
          p_value: 0.03,
          result_text: 'The observed proportion of 0.667 differs significantly from 0.5.',
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Binomial test' })).toBeInTheDocument())
    expect(screen.getByText('Significant')).toBeInTheDocument()
    expect(screen.getByText('Observed proportion differs from expected.')).toBeInTheDocument()
    expect(screen.getByText('The observed proportion of 0.667 differs significantly from 0.5.')).toBeInTheDocument()
  })

  it('shows the error message from the backend on failure', async () => {
    installSession()
    server.use(
      http.post('/api/categorical/binomial', () =>
        HttpResponse.json({ detail: 'Column is not binary' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<CategoricalTestsPanel />)
    await user.click(screen.getByRole('button', { name: /run test/i }))

    await waitFor(() => expect(screen.getByText('Column is not binary')).toBeInTheDocument())
  })
})
