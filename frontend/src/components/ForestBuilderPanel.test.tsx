import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import { clearSession, installSession, makeSession } from '../test/testUtils'
import ForestBuilderPanel from './ForestBuilderPanel'

afterEach(() => clearSession())

describe('ForestBuilderPanel', () => {
  it('renders the empty-state illustration with no rows entered', () => {
    clearSession()
    render(<ForestBuilderPanel />)
    expect(screen.getByText('Interactive Forest Plot Builder')).toBeInTheDocument()
    expect(screen.queryByTestId('plotly-mock')).not.toBeInTheDocument()
  })

  it('manually entering a valid row renders the forest plot', async () => {
    clearSession()
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    const rowsTable = screen.getByText('Rows').closest('div') as HTMLElement
    const table = within(rowsTable.parentElement as HTMLElement).getByRole('table')
    const firstRow = within(table).getAllByRole('row')[1] // header + first data row

    await user.type(within(firstRow).getByPlaceholderText('Label'), 'Model A')
    const numberInputs = within(firstRow).getAllByRole('spinbutton')
    // Order: Est, CI low, CI high, p
    await user.type(numberInputs[0], '2.03')
    await user.type(numberInputs[1], '1.02')
    await user.type(numberInputs[2], '4.03')
    await user.type(numberInputs[3], '0.04')

    expect(screen.queryByText('Interactive Forest Plot Builder')).not.toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
    expect(screen.getByText('1 of 1 valid')).toBeInTheDocument()
  })

  it('loading a preset populates the rows table and renders the plot', async () => {
    clearSession()
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    await user.click(screen.getByRole('button', { name: 'Sensitivity — model specifications' }))

    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
    expect(screen.getByText('6 of 6 valid')).toBeInTheDocument()

    // "Clear all" resets back to the empty state
    await user.click(screen.getByRole('button', { name: '✕ Clear all' }))
    expect(screen.getByText('Interactive Forest Plot Builder')).toBeInTheDocument()
    expect(screen.getByText('0 of 1 valid')).toBeInTheDocument()
  })

  it('adding, reordering, and deleting rows updates the valid-row count', async () => {
    clearSession()
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    await user.click(screen.getByRole('button', { name: 'Multiple endpoints / time horizons' }))
    expect(screen.getByText('5 of 5 valid')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '+ Add row' }))
    expect(screen.getByText('5 of 6 valid')).toBeInTheDocument()

    // Delete the newly added (invalid, last) row via its own row's delete button
    const rowsTable = screen.getByText('Rows').closest('div') as HTMLElement
    const table = within(rowsTable.parentElement as HTMLElement).getByRole('table')
    const dataRows = within(table).getAllByRole('row').slice(1) // skip header
    const lastRow = dataRows[dataRows.length - 1]
    await user.click(within(lastRow).getByTitle('Delete'))
    expect(screen.getByText('5 of 5 valid')).toBeInTheDocument()

    // Move the first row down — row count / validity unaffected, but the
    // reorder handler should not throw and the table stays at 5 rows.
    const rowsAfterDelete = within(table).getAllByRole('row').slice(1)
    await user.click(within(rowsAfterDelete[0]).getByTitle('Move down'))
    expect(screen.getByText('5 of 5 valid')).toBeInTheDocument()
  })

  it('bulk paste (CSV) parses rows and skips a non-numeric header line', async () => {
    clearSession()
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    await user.click(screen.getByRole('button', { name: '📋 Paste rows' }))
    const textarea = screen.getByPlaceholderText(/Unadjusted, 2.03, 1.02, 4.03, 0.04/)
    await user.type(
      textarea,
      'label,est,ci_low,ci_high,p{Enter}Unadjusted,2.03,1.02,4.03,0.04{Enter}Adjusted,1.27,0.61,2.63,0.52',
    )
    await user.click(screen.getByRole('button', { name: 'Apply' }))

    expect(screen.getByText('2 of 2 valid')).toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })

  it('Load from Active Dataset: with a session, auto-maps columns and loads preview rows on click', async () => {
    installSession(
      makeSession({
        columns: [
          { name: 'study', dtype: 'object', kind: 'categorical' },
          { name: 'hr', dtype: 'float64', kind: 'numeric' },
          { name: 'ci_low', dtype: 'float64', kind: 'numeric' },
          { name: 'ci_high', dtype: 'float64', kind: 'numeric' },
        ],
        preview: [
          { study: 'Model A', hr: 1.5, ci_low: 1.1, ci_high: 2.0 },
          { study: 'Model B', hr: 0.9, ci_low: 0.6, ci_high: 1.3 },
        ],
      }),
    )
    const user = userEvent.setup()
    render(<ForestBuilderPanel />)

    expect(screen.getByText('test.csv')).toBeInTheDocument()
    const loadBtn = screen.getByRole('button', { name: /Load Dataset Rows/ })
    expect(loadBtn).toBeEnabled()
    await user.click(loadBtn)

    expect(screen.getByText('2 of 2 valid')).toBeInTheDocument()
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })
})
