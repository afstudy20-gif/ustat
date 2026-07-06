import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import DataDictionaryPanel from './DataDictionaryPanel'

afterEach(() => clearSession())

describe('DataDictionaryPanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<DataDictionaryPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('lists all session columns as dictionary rows', () => {
    installSession()
    render(<DataDictionaryPanel />)
    expect(screen.getByText('AGE')).toBeInTheDocument()
    expect(screen.getByText('LDL')).toBeInTheDocument()
    expect(screen.getByText('GROUP')).toBeInTheDocument()
    expect(
      screen.getByText(
        (_, el) => el?.tagName === 'P' && /4 variables/.test(el?.textContent ?? '') && /3 observations/.test(el?.textContent ?? ''),
      ),
    ).toBeInTheDocument()
  })

  it('lets the user edit a label and role for a column', async () => {
    installSession()
    const user = userEvent.setup()
    render(<DataDictionaryPanel />)

    const row = screen.getByText('AGE').closest('tr')!
    const labelInput = within(row).getByPlaceholderText(/Variable label/)
    await user.type(labelInput, 'Age in years')
    expect(labelInput).toHaveValue('Age in years')

    const roleSelect = within(row).getByRole('combobox')
    await user.selectOptions(roleSelect, 'covariate')
    expect(roleSelect).toHaveValue('covariate')
  })

  it('saves metadata and shows the saved confirmation on success', async () => {
    installSession()
    server.use(
      http.post('/api/sessions/test-session/metadata', () => HttpResponse.json({ ok: true })),
    )

    const user = userEvent.setup()
    render(<DataDictionaryPanel />)

    await user.click(screen.getByRole('button', { name: /save metadata/i }))

    await waitFor(() => expect(screen.getByRole('button', { name: /saved/i })).toBeInTheDocument())
  })

  it('opens the value-labels editor and loads unique values for a column', async () => {
    installSession()
    server.use(
      http.get('/api/compute/test-session/unique/GROUP', () =>
        HttpResponse.json({ values: ['A', 'B'] }),
      ),
    )

    const user = userEvent.setup()
    render(<DataDictionaryPanel />)

    const row = screen.getByText('GROUP').closest('tr')!
    await user.click(within(row).getByRole('button', { name: /edit/i }))

    await waitFor(() => expect(screen.getByText(/Value labels for/)).toBeInTheDocument())
    expect(screen.getByText('A')).toBeInTheDocument()
    expect(screen.getByText('B')).toBeInTheDocument()
  })
})
