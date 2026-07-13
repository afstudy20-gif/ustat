import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import PSMPanel from './PSMPanel'

afterEach(() => clearSession())

const PSM_RESULT = {
  balance_achieved: true,
  n_total: 100,
  n_treated: 40,
  n_control: 60,
  n_matched_pairs: 38,
  n_matched_controls: 38,
  n_unmatched: 2,
  n_trimmed_common_support: 0,
  caliper_used: 0.04,
  caliper_scale: 'logit',
  smd_before: { AGE: 0.4, LDL: 0.3 },
  smd_after: { AGE: 0.06, LDL: 0.07 },
  avg_smd_before: 0.35,
  avg_smd_after: 0.065,
  reduction_pct: 81.4,
  variance_ratio_after: { AGE: 1.05, LDL: 0.95 },
  variance_ratio_before: { AGE: 1.6, LDL: 1.7 },
  ks_p_after: { AGE: 0.5, LDL: 0.4 },
  ps_distribution: {
    treated_unmatched: [0.3, 0.4],
    control_unmatched: [0.2, 0.3],
    treated_matched: [0.3, 0.4],
    control_matched: [0.2, 0.3],
  },
}

async function selectCovariates(user: ReturnType<typeof userEvent.setup>) {
  // "AGE"/"LDL" checkboxes also appear in the "Exact match strata" list, so scope to
  // the Covariates (Confounders) panel specifically.
  const covariatesPanel = screen.getByText('Covariates (Confounders)').closest('div.panel') as HTMLElement
  await user.click(within(covariatesPanel).getByRole('checkbox', { name: 'AGE' }))
  await user.click(within(covariatesPanel).getByRole('checkbox', { name: 'LDL' }))
}

describe('PSMPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<PSMPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('runs greedy 1:1 matching (default) and renders balance summary + SMD table', async () => {
    installSession()
    server.use(http.post('/api/models/psm', () => HttpResponse.json(PSM_RESULT)))

    const user = userEvent.setup()
    render(<PSMPanel />)

    await selectCovariates(user)

    const runButton = screen.getByRole('button', { name: /run psm/i })
    expect(runButton).toBeEnabled()
    await user.click(runButton)

    await waitFor(() => expect(screen.getByText(/Balance achieved/i)).toBeInTheDocument())
    expect(screen.getByText(/Matched 38 treated : 38 control pairs/)).toBeInTheDocument()
    expect(screen.getByText('SMD Balance Table')).toBeInTheDocument()
    const rows = screen.getAllByRole('row')
    expect(rows.length).toBeGreaterThanOrEqual(3)
  })

  it('runs optimal matching with a 1:2 ratio and renders results', async () => {
    installSession()
    let capturedBody: Record<string, unknown> | null = null
    server.use(
      http.post('/api/models/psm', async ({ request }) => {
        capturedBody = (await request.json()) as Record<string, unknown>
        return HttpResponse.json({ ...PSM_RESULT, n_matched_controls: 76 })
      }),
    )

    const user = userEvent.setup()
    render(<PSMPanel />)

    await selectCovariates(user)

    // Switch matching method to Optimal
    await user.click(screen.getByRole('button', { name: 'Optimal' }))

    // Locate the ratio <select> by its sibling label text "Ratio"
    const ratioLabel = screen.getByText('Ratio')
    const ratioContainer = ratioLabel.closest('div')!
    const ratioSelectEl = within(ratioContainer).getByRole('combobox')
    await user.selectOptions(ratioSelectEl, '2')

    await user.click(screen.getByRole('button', { name: /run psm/i }))

    await waitFor(() => expect(screen.getByText(/Balance achieved/i)).toBeInTheDocument())
    expect(screen.getByText(/Matched 38 treated : 76 control pairs/)).toBeInTheDocument()

    expect(capturedBody).not.toBeNull()
    expect(capturedBody!.matching_method).toBe('optimal')
    expect(capturedBody!.ratio).toBe(2)
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/models/psm', () =>
        HttpResponse.json({ detail: 'Treatment column must be binary (0/1)' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<PSMPanel />)

    await selectCovariates(user)
    await user.click(screen.getByRole('button', { name: /run psm/i }))

    await waitFor(() =>
      expect(screen.getByText('Treatment column must be binary (0/1)')).toBeInTheDocument(),
    )
  })

  it('disables Run PSM until at least one covariate is selected', () => {
    installSession()
    render(<PSMPanel />)
    expect(screen.getByRole('button', { name: /run psm/i })).toBeDisabled()
  })
})
