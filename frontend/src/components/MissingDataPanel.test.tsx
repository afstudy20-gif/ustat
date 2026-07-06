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
