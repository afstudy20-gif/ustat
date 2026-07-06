import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse, delay } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import CodePanel from './CodePanel'

afterEach(() => clearSession())

const enabledStatus = {
  enabled: true,
  max_timeout_s: 60,
  max_code_bytes: 102400,
  rate_limit_per_min: 6,
  rate_limit_per_hour: 30,
}

describe('CodePanel', () => {
  it('shows an upload prompt without an active session', () => {
    clearSession()
    server.use(http.get('/api/code/status', () => HttpResponse.json(enabledStatus)))
    render(<CodePanel />)
    expect(screen.getByText(/upload a dataset to access the code panel/i)).toBeInTheDocument()
  })

  it('renders the Python sandbox editor with a session and status enabled', async () => {
    installSession()
    server.use(http.get('/api/code/status', () => HttpResponse.json(enabledStatus)))
    render(<CodePanel />)
    expect(screen.getByRole('button', { name: /run/i })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: /run/i })).toBeEnabled())
  })

  it('disables Run and shows a message when the code runner is disabled on the server', async () => {
    installSession()
    server.use(
      http.get('/api/code/status', () =>
        HttpResponse.json({ ...enabledStatus, enabled: false }),
      ),
    )
    render(<CodePanel />)
    await waitFor(() =>
      expect(screen.getByText(/code runner is disabled on this server/i)).toBeInTheDocument(),
    )
    expect(screen.getByRole('button', { name: /run/i })).toBeDisabled()
  })

  it('runs code and renders stdout in the Console tab on success', async () => {
    installSession()
    server.use(
      http.get('/api/code/status', () => HttpResponse.json(enabledStatus)),
      http.post('/api/code/run', () =>
        HttpResponse.json({
          stdout: 'hello from sandbox\n',
          stderr: '',
          figures: [],
          exit_code: 0,
          time_used_s: 0.42,
          error: null,
          timed_out: false,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CodePanel />)
    await waitFor(() => expect(screen.getByRole('button', { name: /run/i })).toBeEnabled())

    await user.click(screen.getByRole('button', { name: /run/i }))

    await waitFor(() => expect(screen.getByText('hello from sandbox')).toBeInTheDocument())
    expect(screen.getByText(/exit=0/)).toBeInTheDocument()
  })

  it('shows figures tab and switches to it automatically when figures are returned', async () => {
    installSession()
    const onePxPng =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
    server.use(
      http.get('/api/code/status', () => HttpResponse.json(enabledStatus)),
      http.post('/api/code/run', () =>
        HttpResponse.json({
          stdout: '',
          stderr: '',
          figures: [onePxPng],
          exit_code: 0,
          time_used_s: 0.1,
          error: null,
          timed_out: false,
        }),
      ),
    )

    const user = userEvent.setup()
    render(<CodePanel />)
    await waitFor(() => expect(screen.getByRole('button', { name: /run/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /run/i }))

    await waitFor(() => expect(screen.getByAltText('figure 1')).toBeInTheDocument())
  })

  it('shows the backend error message when the run fails', async () => {
    installSession()
    server.use(
      http.get('/api/code/status', () => HttpResponse.json(enabledStatus)),
      http.post('/api/code/run', () =>
        HttpResponse.json({ detail: 'Code exceeded memory limit' }, { status: 400 }),
      ),
    )

    const user = userEvent.setup()
    render(<CodePanel />)
    await waitFor(() => expect(screen.getByRole('button', { name: /run/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /run/i }))

    await waitFor(() => expect(screen.getByText('Code exceeded memory limit')).toBeInTheDocument())
  })

  it('supports cancelling an in-flight run via the Stop button', async () => {
    installSession()
    server.use(
      http.get('/api/code/status', () => HttpResponse.json(enabledStatus)),
      http.post('/api/code/run', async () => {
        await delay(5000)
        return HttpResponse.json({
          stdout: 'should not appear',
          stderr: '',
          figures: [],
          exit_code: 0,
          time_used_s: 5,
          error: null,
          timed_out: false,
        })
      }),
    )

    const user = userEvent.setup()
    render(<CodePanel />)
    await waitFor(() => expect(screen.getByRole('button', { name: /run/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /run/i }))

    const stopButton = await screen.findByRole('button', { name: /stop/i })
    await user.click(stopButton)

    await waitFor(() => expect(screen.getByRole('button', { name: /^run$/i })).toBeInTheDocument())
    expect(screen.queryByText('should not appear')).not.toBeInTheDocument()
  })
})
