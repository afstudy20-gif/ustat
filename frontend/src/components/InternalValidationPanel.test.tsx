import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import InternalValidationPanel from './InternalValidationPanel'

afterEach(() => clearSession())

describe('InternalValidationPanel', () => {
  it('renders the tab bar even without an active session', () => {
    clearSession()
    render(<InternalValidationPanel />)
    expect(screen.getByRole('button', { name: /internal \(bootstrap \+ cv\)/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /external \(logistic\)/i })).toBeInTheDocument()
  })

  it('defaults to the Internal tab with Logistic model type', () => {
    installSession()
    render(<InternalValidationPanel />)
    expect(screen.getByRole('button', { name: 'Logistic' })).toBeInTheDocument()
    expect(screen.getByText('Outcome (binary 0/1)')).toBeInTheDocument()
  })

  it('disables Run until outcome and predictors are chosen', () => {
    installSession()
    render(<InternalValidationPanel />)
    expect(screen.getByRole('button', { name: /run internal validation/i })).toBeDisabled()
  })

  it('runs internal validation (logistic) and renders discrimination tiles', async () => {
    installSession()
    server.use(
      http.post('/api/model_diagnostics/model_validation', () =>
        HttpResponse.json({
          interpretation: 'Modest overfitting detected',
          n: 100, n_predictors: 2, n_boot: 200,
          apparent: { auc: 0.82, calibration_slope: 1.0, brier: 0.15 },
          optimism: { auc: 0.05 },
          corrected: { auc: 0.77, calibration_slope: 0.9 },
          cv: { auc: 0.76, calibration_slope: 0.88, brier: 0.16, folds: 5 },
          overfit_gap: 0.05,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)

    const outcomeSelect = screen.getByText('Outcome (binary 0/1)').closest('div')!.querySelector('select')!
    await user.selectOptions(outcomeSelect, 'DM')
    const predictorCheckbox = screen.getByRole('checkbox', { name: 'AGE' })
    await user.click(predictorCheckbox)

    await user.click(screen.getByRole('button', { name: /run internal validation/i }))

    await waitFor(() => expect(screen.getByText('Modest overfitting detected')).toBeInTheDocument())
    expect(screen.getByText('0.820')).toBeInTheDocument()
    expect(screen.getByText('0.770')).toBeInTheDocument()
  })

  it('switches to the External tab and runs external validation', async () => {
    installSession()
    server.use(
      http.post('/api/model_diagnostics/external_validation_logistic', () =>
        HttpResponse.json({
          result_text: 'Calibration acceptable in validation cohort',
          n: 50,
          discrimination: { auc: 0.79, auc_ci: [0.7, 0.88], se: 0.05 },
          calibration: {
            slope: 0.95, intercept: 0.02, oe_ratio: 1.01,
            hosmer_lemeshow: { chi2: 5.1, df: 8, p: 0.75 },
            brier: 0.14, acceptable: true,
          },
          calibration_plot: [{ pred: 0.2, obs: 0.22, n: 20 }],
          dev_vs_val: { auc_drop: 0.03, slope_shift: -0.05 },
        }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    await user.click(screen.getByRole('button', { name: /external \(logistic\)/i }))

    const outcomeSelect = screen.getByText('Outcome (binary 0/1)').closest('div')!.querySelector('select')!
    await user.selectOptions(outcomeSelect, 'DM')
    const probSelect = screen.getByText('Predicted probability column (0–1)').closest('div')!.querySelector('select')!
    await user.selectOptions(probSelect, 'AGE')

    await user.click(screen.getByRole('button', { name: /run external validation/i }))

    await waitFor(() => expect(screen.getByText('Calibration acceptable in validation cohort')).toBeInTheDocument())
    expect(screen.getByText('0.790')).toBeInTheDocument()
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/model_diagnostics/model_validation', () =>
        HttpResponse.json({ detail: 'Too few events for bootstrap' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<InternalValidationPanel />)
    const outcomeSelect = screen.getByText('Outcome (binary 0/1)').closest('div')!.querySelector('select')!
    await user.selectOptions(outcomeSelect, 'DM')
    await user.click(screen.getByRole('checkbox', { name: 'AGE' }))
    await user.click(screen.getByRole('button', { name: /run internal validation/i }))

    await waitFor(() => expect(screen.getByText('Too few events for bootstrap')).toBeInTheDocument())
  })
})
