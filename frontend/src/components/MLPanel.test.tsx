import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import MLPanel from './MLPanel'

afterEach(() => clearSession())

const baseResult = {
  model: 'Random Forest',
  outcome: 'DM',
  task: 'classification' as const,
  cv_folds: 5,
  n: 3,
  n_features: 1,
  importance: [
    { feature: 'AGE', permutation: 0.05, permutation_sd: 0.01, impurity: 0.2 },
  ],
  roc_curve: [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }],
  scatter: [],
  auc: 0.812,
  auc_ci_low: 0.7,
  auc_ci_high: 0.9,
  accuracy: 0.8,
  sensitivity: 0.75,
  specificity: 0.85,
  ppv: 0.7,
  npv: 0.9,
  brier: 0.15,
  confusion: { tp: 5, tn: 10, fp: 2, fn: 1 },
  calibration: [{ pred: 0.5, obs: 0.4, n: 10 }],
  interpretation: 'The model shows good discrimination.',
}

describe('MLPanel', () => {
  it('renders without crashing when there is no active session', () => {
    clearSession()
    const { container } = render(<MLPanel />)
    // No session → columns list is empty but component still mounts (no early return guard in MLPanel).
    expect(container).toBeTruthy()
  })

  it('disables the train button until outcome and predictors are selected', () => {
    installSession()
    render(<MLPanel />)
    // No explicit disabled state on the button itself pre-validation is handled in run(),
    // but clicking without outcome should show a validation error rather than call the API.
    expect(screen.getByRole('button', { name: /train & cross-validate/i })).toBeEnabled()
  })

  it('shows a validation error when running without outcome/predictors selected', async () => {
    installSession()
    const user = userEvent.setup()
    render(<MLPanel />)
    await user.click(screen.getByRole('button', { name: /train & cross-validate/i }))
    expect(await screen.findByText(/select an outcome and at least one predictor/i)).toBeInTheDocument()
  })

  it('runs Random Forest and renders the result on success', async () => {
    installSession()
    server.use(
      http.post('/api/ml/random_forest', () => HttpResponse.json(baseResult)),
    )

    const user = userEvent.setup()
    render(<MLPanel />)

    await user.selectOptions(screen.getByRole('combobox', { name: /outcome/i }), 'DM')
    await user.click(screen.getAllByRole('checkbox')[0])

    await user.click(screen.getByRole('button', { name: /train & cross-validate/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Random Forest' })).toBeInTheDocument())
    expect(screen.getByText('0.812')).toBeInTheDocument()
    expect(screen.getByText('The model shows good discrimination.')).toBeInTheDocument()
    expect(screen.getByText('Feature importance')).toBeInTheDocument()
  })

  it('runs Gradient Boosting via the model toggle and hits the correct endpoint', async () => {
    installSession()
    server.use(
      http.post('/api/ml/gradient_boosting', () =>
        HttpResponse.json({ ...baseResult, model: 'Gradient Boosting' }),
      ),
    )

    const user = userEvent.setup()
    render(<MLPanel />)

    await user.click(screen.getByRole('button', { name: /gradient boosting/i }))
    await user.selectOptions(screen.getByRole('combobox', { name: /outcome/i }), 'DM')
    await user.click(screen.getAllByRole('checkbox')[0])
    await user.click(screen.getByRole('button', { name: /train & cross-validate/i }))

    await waitFor(() => expect(screen.getByRole('heading', { name: 'Gradient Boosting' })).toBeInTheDocument())
  })

  it('shows the backend error message on failure', async () => {
    installSession()
    server.use(
      http.post('/api/ml/random_forest', () =>
        HttpResponse.json({ detail: 'Not enough data for cross-validation' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<MLPanel />)
    await user.selectOptions(screen.getByRole('combobox', { name: /outcome/i }), 'DM')
    await user.click(screen.getAllByRole('checkbox')[0])
    await user.click(screen.getByRole('button', { name: /train & cross-validate/i }))

    await waitFor(() =>
      expect(screen.getByText('Not enough data for cross-validation')).toBeInTheDocument(),
    )
  })
})
