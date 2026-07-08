import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import type { ColMeta } from '../store'
import { server } from '../test/server'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import MissingDataPanel from './MissingDataPanel'

afterEach(() => clearSession())

// Session with actual missing values so the Overview table renders rows
// (installSession()'s default preview has no nulls/blanks).
const columnsWithMissing: ColMeta[] = [
  { name: 'AGE', dtype: 'float64', kind: 'numeric' },
  { name: 'LDL', dtype: 'float64', kind: 'numeric' },
  { name: 'GROUP', dtype: 'object', kind: 'categorical' },
]

function installMissingSession() {
  installSession(
    makeSession({
      columns: columnsWithMissing,
      preview: [
        { AGE: 55, LDL: null, GROUP: 'A' },
        { AGE: null, LDL: 140, GROUP: 'B' },
        { AGE: 48, LDL: 110, GROUP: '' },
      ],
    }),
  )
}

describe('MissingDataPanel', () => {
  it('shows an upload prompt without an active session', () => {
    clearSession()
    render(<MissingDataPanel />)
    expect(screen.getByText(/upload data first/i)).toBeInTheDocument()
  })

  it('shows the all-clear message when no columns have missing values', () => {
    installSession()
    render(<MissingDataPanel />)
    expect(screen.getByText(/no missing values detected/i)).toBeInTheDocument()
  })

  it('lists columns with missing data in the Overview sub-tab', () => {
    installMissingSession()
    render(<MissingDataPanel />)
    expect(screen.getByRole('tab', { name: /missing data overview/i })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    const table = screen.getAllByRole('table')[0]
    expect(within(table).getByText('AGE')).toBeInTheDocument()
    expect(within(table).getByText('LDL')).toBeInTheDocument()
    expect(within(table).getByText('GROUP')).toBeInTheDocument()
  })

  it('runs Little\'s MCAR test + missingness diagnostics after selecting columns', async () => {
    installMissingSession()
    server.use(
      http.post('/api/compute/test-session/missing_diagnostics', () =>
        HttpResponse.json({
          columns: [
            { name: 'AGE', n_missing: 1, pct: 33.3, kind: 'numeric', is_numeric: true, depends_on: ['LDL'], likely: 'MAR' },
            { name: 'LDL', n_missing: 1, pct: 33.3, kind: 'numeric', is_numeric: true, depends_on: [], likely: 'MCAR' },
          ],
          overall_hint: 'Some dependence detected.',
          recommendation: 'Use MICE.',
          any_mar: true,
        }),
      ),
      http.post('/api/missing_data/mcar_test', () =>
        HttpResponse.json({ statistic: 4.21, df: 2, p: 0.12, significant: false }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    const table = screen.getAllByRole('table')[0]
    const ageRow = within(table).getByText('AGE').closest('tr')!
    const ldlRow = within(table).getByText('LDL').closest('tr')!
    await user.click(within(ageRow).getByRole('checkbox'))
    await user.click(within(ldlRow).getByRole('checkbox'))

    await user.click(screen.getByRole('button', { name: /analyze missingness/i }))

    await waitFor(() => expect(screen.getByText(/little's mcar test/i)).toBeInTheDocument())
    expect(
      screen.getAllByText(
        (_, el) => el?.tagName === 'DIV' && (el?.textContent ?? '').includes('χ²=4.21, df=2, p=0.120.'),
      ).length,
    ).toBeGreaterThan(0)
    expect(screen.getByText(/Some dependence detected\./)).toBeInTheDocument()
    expect(screen.getByText(/Use MICE\./)).toBeInTheDocument()
    expect(screen.getByText(/missingness related to LDL/)).toBeInTheDocument()
  })

  it('shows a diagnostics error from the backend', async () => {
    installMissingSession()
    server.use(
      http.post('/api/compute/test-session/missing_diagnostics', () =>
        HttpResponse.json({ detail: 'Diagnostics failed' }, { status: 500 }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)
    await user.click(screen.getAllByRole('checkbox')[0])
    await user.click(screen.getByRole('button', { name: /analyze missingness/i }))

    await waitFor(() => expect(screen.getByText('Diagnostics failed')).toBeInTheDocument())
  })

  it('previews reference dataset imputation', async () => {
    installMissingSession()
    server.use(
      http.post('/api/missing_data/external_impute_reference_columns', () =>
        HttpResponse.json({
          n_rows: 2,
          columns: [
            { name: 'age', dtype: 'int64', kind: 'numeric', n_missing: 0 },
            { name: 'ldl', dtype: 'int64', kind: 'numeric', n_missing: 0 },
            { name: 'REFERENCE_ONLY', dtype: 'object', kind: 'categorical', n_missing: 0 },
          ],
        }),
      ),
      http.post('/api/missing_data/external_impute_preview', async ({ request }) => {
        const fd = await request.formData()
        expect(fd.get('target')).toBe('LDL')
        expect(fd.get('reference_target')).toBe('ldl')
        expect(fd.get('predictors')).toBe(JSON.stringify(['age']))
        expect(fd.get('predictor_mappings')).toBe(JSON.stringify({ age: 'AGE' }))
        expect(fd.get('file')).toBeTruthy()
        return HttpResponse.json({
          target: 'LDL',
          reference_target: 'ldl',
          predictors: ['AGE'],
          reference_predictors: ['age'],
          method: 'PMM',
          mechanism: 'unknown',
          n_missing_target: 1,
          n_imputed: 1,
          reference_rows: 2,
          reference_complete_rows: 2,
          preview_rows: [{ row_index: 0, imputed_value: 128, predictors_missing: 0 }],
          result_text: "1 missing value(s) in 'LDL' were imputed using 1 predictor(s).",
        })
      }),
      http.post('/api/missing_data/external_impute_transfer', async ({ request }) => {
        const body = await request.json() as {
          session_id: string;
          target: string;
          preview_rows: Array<{ row_index: number; imputed_value: unknown }>;
        }
        expect(body.session_id).toBe('test-session')
        expect(body.target).toBe('LDL')
        expect(body.preview_rows).toEqual([{ row_index: 0, imputed_value: 128 }])
        return HttpResponse.json({
          target: 'LDL',
          n_imputed: 1,
          applied: true,
          result_text: "1 previewed value(s) were transferred into 'LDL'.",
        })
      }),
      http.get('/api/stats/test-session/refresh', () =>
        HttpResponse.json({
          columns: columnsWithMissing,
          preview: [
            { AGE: 55, LDL: 128, GROUP: 'A' },
            { AGE: null, LDL: 140, GROUP: 'B' },
            { AGE: 48, LDL: 110, GROUP: '' },
          ],
        }),
      ),
    )

    const user = userEvent.setup()
    render(<MissingDataPanel />)

    await user.click(screen.getByRole('tab', { name: /reference imputation/i }))
    await user.selectOptions(screen.getByLabelText(/current missing target/i), 'LDL')
    await user.upload(
      screen.getByLabelText(/reference dataset/i),
      new File(['age,ldl\n55,128\n61,140\n'], 'reference.csv', { type: 'text/csv' }),
    )
    await waitFor(() => expect(screen.getByLabelText(/reference target match/i)).toHaveValue('ldl'))
    await waitFor(() => expect(screen.getAllByText('REFERENCE_ONLY').length).toBeGreaterThan(0))
    expect(screen.getByDisplayValue('AGE')).toBeInTheDocument()
    const agePredictor = screen.getByLabelText('age')
    await user.click(agePredictor)
    await user.click(screen.getByRole('button', { name: /preview target estimates/i }))

    await waitFor(() => expect(screen.getByText(/1 missing value/)).toBeInTheDocument())
    expect(screen.getByText('128')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /transfer data/i })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: /transfer data/i }))
    await waitFor(() => expect(screen.getByText(/1 value\(s\) transferred into LDL/)).toBeInTheDocument())
  })

  it('switches to the Data Cleaning sub-tab', async () => {
    installMissingSession()
    const user = userEvent.setup()
    render(<MissingDataPanel />)

    const cleaningTab = screen.getByRole('tab', { name: /data cleaning/i })
    await user.click(cleaningTab)
    expect(cleaningTab).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: /missing data overview/i })).toHaveAttribute(
      'aria-selected',
      'false',
    )
  })
})
