import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import { useStore } from '../store'
import ComputePanel from './ComputePanel'

afterEach(() => clearSession())
beforeEach(() => localStorage.clear())

const computeSession = () =>
  makeSession({
    columns: [
      { name: 'WEIGHT', dtype: 'float64', kind: 'numeric' },
      { name: 'HEIGHT', dtype: 'float64', kind: 'numeric' },
    ],
    preview: [
      { WEIGHT: 70, HEIGHT: 170 },
      { WEIGHT: 80, HEIGHT: 180 },
      { WEIGHT: 60, HEIGHT: 160 },
    ],
  })

describe('ComputePanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<ComputePanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('Formula tab: applies a formula and the new column appears in the session', async () => {
    installSession(computeSession())
    server.use(
      http.post('/api/compute/test-session/formula', () =>
        HttpResponse.json({
          name: 'BMI',
          dtype: 'float64',
          kind: 'numeric',
          preview_values: [24.2, 24.7, 23.4],
          n_computed: 3,
          n_missing: 0,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ComputePanel />)

    const newColInput = screen.getByPlaceholderText('e.g. BMI')
    await user.type(newColInput, 'BMI')
    const formulaInput = screen.getByPlaceholderText(/Weight \/ \(\(Height/)
    await user.type(formulaInput, 'WEIGHT / ((HEIGHT / 100) ** 2)')

    const applyBtn = screen.getByRole('button', { name: 'Apply Formula' })
    expect(applyBtn).toBeEnabled()
    await user.click(applyBtn)

    await waitFor(() =>
      expect(
        screen.getByText((_, el) => el?.textContent === 'BMI created — 3 values computed'),
      ).toBeInTheDocument(),
    )
    // The new column was mirrored into the store's session (addSessionColumn).
    const cols = useStore.getState().session?.columns.map((c) => c.name)
    expect(cols).toContain('BMI')
  })

  it('Formula tab: shows the backend error message on an invalid formula', async () => {
    installSession(computeSession())
    server.use(
      http.post('/api/compute/test-session/formula', () =>
        HttpResponse.json({ detail: "Unknown column 'FOO' in formula" }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<ComputePanel />)

    await user.type(screen.getByPlaceholderText('e.g. BMI'), 'BadCol')
    await user.type(screen.getByPlaceholderText(/Weight \/ \(\(Height/), 'FOO * 2')
    await user.click(screen.getByRole('button', { name: 'Apply Formula' }))

    await waitFor(() =>
      expect(screen.getByText("Unknown column 'FOO' in formula")).toBeInTheDocument(),
    )
  })

  it('Clinical tab: BMI calculator computes and creates a new column', async () => {
    installSession(computeSession())
    server.use(
      http.post('/api/compute/test-session/clinical/bmi', () =>
        HttpResponse.json({
          name: 'BMI',
          dtype: 'float64',
          kind: 'numeric',
          preview_values: [24.2, 24.7, 23.4],
          n_computed: 3,
          n_missing: 0,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<ComputePanel />)

    await user.click(screen.getByRole('button', { name: /Clinical/ }))
    await user.click(screen.getByRole('button', { name: /BMI.*Body Mass Index/ }))

    const calcBtn = screen.getByRole('button', { name: 'Calculate BMI' })
    expect(calcBtn).toBeEnabled()
    await user.click(calcBtn)

    await waitFor(() =>
      expect(
        screen.getByText((_, el) => el?.textContent === 'BMI created — 3 values computed'),
      ).toBeInTheDocument(),
    )
    const cols = useStore.getState().session?.columns.map((c) => c.name)
    expect(cols).toContain('BMI')
  })
})
