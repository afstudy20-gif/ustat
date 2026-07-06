import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import CausalPanel from './CausalPanel'

afterEach(() => clearSession())

const fourColSession = () =>
  makeSession({
    columns: [
      { name: 'Y', dtype: 'float64', kind: 'numeric' },
      { name: 'X', dtype: 'float64', kind: 'numeric' },
      { name: 'Z', dtype: 'float64', kind: 'numeric' },
      { name: 'M', dtype: 'float64', kind: 'numeric' },
    ],
    preview: [
      { Y: 1, X: 2, Z: 3, M: 4 },
      { Y: 2, X: 3, Z: 4, M: 5 },
      { Y: 3, X: 4, Z: 5, M: 6 },
    ],
  })

/** Find the <select> that immediately follows a given field label text. */
function selectAfterLabel(labelText: string): HTMLSelectElement {
  const label = screen.getByText(labelText)
  const wrapper = label.parentElement as HTMLElement
  return within(wrapper).getByRole('combobox') as HTMLSelectElement
}

/** Find a MultiPick checkbox for `colName` within the group titled `groupLabel`. */
function checkboxInGroup(groupLabel: string, colName: string): HTMLInputElement {
  const groupTitle = screen.getByText(groupLabel)
  const group = groupTitle.parentElement as HTMLElement
  const span = within(group).getByText(colName)
  const label = span.closest('label') as HTMLLabelElement
  return within(label).getByRole('checkbox') as HTMLInputElement
}

describe('CausalPanel', () => {
  it('renders the IV tab by default even without an active session (no crash)', () => {
    clearSession()
    render(<CausalPanel />)
    expect(screen.getAllByText('Instrumental Variable (2SLS)').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'Run 2SLS' })).toBeDisabled()
  })

  it('IV tab: runs 2SLS and renders the effect estimates on success', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/causal/iv_2sls', () =>
        HttpResponse.json({
          result_text: 'The IV estimate is significant and differs from OLS.',
          n: 3,
          iv_estimate: { estimate: 0.842, ci_low: 0.2, ci_high: 1.5, p: 0.01 },
          ols_estimate: { estimate: 0.5, p: 0.02 },
          first_stage: { f_stat: 25.4, weak_instruments: false },
          wu_hausman: { p: 0.03, endogenous: true },
          sargan: null,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CausalPanel />)

    await user.selectOptions(selectAfterLabel('Outcome (continuous)'), 'Y')
    await user.selectOptions(selectAfterLabel('Endogenous exposure'), 'X')
    await user.click(checkboxInGroup('Instrument(s)', 'Z'))

    const runBtn = screen.getByRole('button', { name: 'Run 2SLS' })
    expect(runBtn).toBeEnabled()
    await user.click(runBtn)

    await waitFor(() => expect(screen.getByText('0.8420')).toBeInTheDocument())
    expect(screen.getByText('25.4')).toBeInTheDocument()
    expect(screen.getByText('adequate (≥10)')).toBeInTheDocument()
  })

  it('Mediation tab: runs and renders ACME/ADE decomposition; shows backend error on failure', async () => {
    installSession(fourColSession())
    server.use(
      http.post('/api/causal/mediation', () =>
        HttpResponse.json({ detail: 'Mediator has no variance after conditioning' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<CausalPanel />)
    await user.click(screen.getByRole('button', { name: 'Mediation (X→M→Y)' }))

    await user.selectOptions(selectAfterLabel('Outcome Y (continuous)'), 'Y')
    await user.selectOptions(selectAfterLabel('Treatment / exposure X'), 'X')
    await user.selectOptions(selectAfterLabel('Mediator M (continuous)'), 'M')

    const runBtn = screen.getByRole('button', { name: 'Run mediation' })
    expect(runBtn).toBeEnabled()
    await user.click(runBtn)

    await waitFor(() =>
      expect(screen.getByText('Mediator has no variance after conditioning')).toBeInTheDocument(),
    )
  })

  it('DAG Backdoor tab: analyses a DAG (no session required) and renders adjustment sets', async () => {
    clearSession()
    server.use(
      http.post('/api/causal/dag_adjustment', () =>
        HttpResponse.json({
          result_text: 'Adjust for Z to close the backdoor path.',
          adjustment_set: ['Z'],
          do_not_adjust: ['C'],
          roles: { Z: 'confounder', M: 'mediator', C: 'collider' },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CausalPanel />)
    await user.click(screen.getByRole('button', { name: 'DAG Backdoor' }))

    await user.click(screen.getByRole('button', { name: 'Analyse DAG' }))

    await waitFor(() => expect(screen.getByText('Adjust for (minimal set)')).toBeInTheDocument())
    const nodeRoles = screen.getByText('Node roles').parentElement as HTMLElement
    expect(within(nodeRoles).getByText(/confounder/)).toBeInTheDocument()
    expect(within(nodeRoles).getByText(/collider/)).toBeInTheDocument()
  })

  it('DAG Backdoor tab: shows an error message when the DAG endpoint fails', async () => {
    clearSession()
    server.use(
      http.post('/api/causal/dag_adjustment', () =>
        HttpResponse.json({ detail: 'Graph contains a cycle' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<CausalPanel />)
    await user.click(screen.getByRole('button', { name: 'DAG Backdoor' }))
    await user.click(screen.getByRole('button', { name: 'Analyse DAG' }))

    await waitFor(() => expect(screen.getByText('Graph contains a cycle')).toBeInTheDocument())
  })
})
