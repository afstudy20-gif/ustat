import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse, delay } from 'msw'
import { afterEach, describe, expect, it } from 'vitest'
import { server } from '../test/server'
import { clearSession, installSession } from '../test/testUtils'
import DescriptivePanel from './DescriptivePanel'

afterEach(() => clearSession())

const numericSummary = {
  type: 'numeric',
  histogram: [
    { bin_start: 40, bin_end: 50, count: 1 },
    { bin_start: 50, bin_end: 60, count: 1 },
    { bin_start: 60, bin_end: 70, count: 1 },
  ],
  raw_values: [55, 62, 48],
  outliers: [],
  normality_deviants: [],
  qq: [
    { x: -1, y: 48 },
    { x: 0, y: 55 },
    { x: 1, y: 62 },
  ],
  n: 3,
  missing: 0,
  display_decimals: 2,
  mean: 55,
  std: 7.02,
  median: 55,
  min: 48,
  max: 62,
  q1: 51.5,
  q3: 58.5,
  iqr: 7,
  whisker_low: 48,
  whisker_high: 62,
  skewness: 0.05,
  kurtosis: -1.2,
  normal: true,
  normality_label: 'Normal',
  normality_test: 'Shapiro-Wilk',
  normality_p: 0.842,
}

const categoricalSummary = {
  type: 'categorical',
  histogram: [],
  qq: [],
  categories: [
    { value: 'A', count: 2, pct: 66.7 },
    { value: 'B', count: 1, pct: 33.3 },
  ],
  n: 3,
  n_categories: 2,
  missing: 0,
}

function mockCommonEndpoints() {
  server.use(
    http.get('/api/stats/test-session/sparklines', () => HttpResponse.json({})),
    http.get('/api/stats/test-session/descriptive', () => HttpResponse.json({})),
  )
}

describe('DescriptivePanel', () => {
  it('renders nothing without an active session', () => {
    clearSession()
    const { container } = render(<DescriptivePanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it('numeric column: loads and displays summary stats, normality test, and n', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', ({ request }) => {
        const url = new URL(request.url)
        expect(url.searchParams.get('column')).toBe('AGE')
        return HttpResponse.json(numericSummary)
      }),
    )

    render(<DescriptivePanel />)

    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    // header row
    await waitFor(() =>
      expect(screen.getByText((_, el) => el?.textContent === 'Continuous · n=3')).toBeInTheDocument(),
    )
    expect(screen.getByText((_, el) => el?.textContent === '· Normal (p=0.842)')).toBeInTheDocument()

    // normality badge box
    expect(screen.getByText('Normal')).toBeInTheDocument()
    expect(screen.getByText((_, el) => el?.textContent === '(Shapiro-Wilk p = 0.842)')).toBeInTheDocument()

    // stats strip
    expect(screen.getByText('Mean')).toBeInTheDocument()
    expect(screen.getAllByText('55.00').length).toBeGreaterThan(0)

    // default chart tab is histogram → plotly mock present
    expect(screen.getByTestId('plotly-mock')).toBeInTheDocument()
  })

  it('categorical column: loads and displays frequency table info', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', ({ request }) => {
        const url = new URL(request.url)
        const column = url.searchParams.get('column')
        if (column === 'GROUP') return HttpResponse.json(categoricalSummary)
        return HttpResponse.json(numericSummary)
      }),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)

    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    await user.click(screen.getByText('GROUP'))

    await waitFor(() =>
      expect(screen.getByText((_, el) => el?.textContent === 'Categorical · n=3')).toBeInTheDocument(),
    )
    expect(screen.getByText('2 categories')).toBeInTheDocument()
    expect(
      screen.getAllByText(
        (_, el) => el?.textContent === '2 categories, n = 3. Report as n (%). Most frequent: A (66.7%).',
      ).length,
    ).toBeGreaterThan(0)
  })

  it('switches between chart tabs (Histogram -> Box Plot -> Q-Q Plot)', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () => HttpResponse.json(numericSummary)),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)

    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    await waitFor(() => expect(screen.getByTestId('plotly-mock')).toBeInTheDocument())

    // Box Plot tab
    await user.click(screen.getByRole('button', { name: 'Box Plot' }))
    await waitFor(() => {
      const plots = screen.getAllByTestId('plotly-mock')
      const boxPlot = plots.find((p) => (p.getAttribute('data-plotly') ?? '').includes('"type":"box"'))
      expect(boxPlot).toBeTruthy()
    })

    // Q-Q Plot tab
    await user.click(screen.getByRole('button', { name: 'Q-Q Plot' }))
    await waitFor(() => {
      const plots = screen.getAllByTestId('plotly-mock')
      const qqPlot = plots.find((p) => (p.getAttribute('data-plotly') ?? '').includes('Reference'))
      expect(qqPlot).toBeTruthy()
    })
  })

  it('shows an error-safe empty state when the summary request fails', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', () =>
        HttpResponse.json({ detail: 'Column not found' }, { status: 500 }),
      ),
    )

    render(<DescriptivePanel />)

    // Loading first
    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    // Summary never resolves successfully -> falls back to "select a column" prompt,
    // no crash and no stale data rendered.
    await waitFor(() =>
      expect(screen.getByText('Select a column to view distribution')).toBeInTheDocument(),
    )
  })

  it('a slow stale request cannot overwrite a faster newer selection (race guard)', async () => {
    installSession()
    mockCommonEndpoints()
    server.use(
      http.get('/api/stats/test-session/column_summary', async ({ request }) => {
        const url = new URL(request.url)
        const column = url.searchParams.get('column')
        if (column === 'AGE') {
          // AGE was clicked FIRST but resolves LAST.
          await delay(50)
          return HttpResponse.json(numericSummary)
        }
        // GROUP was clicked SECOND but resolves FIRST.
        return HttpResponse.json(categoricalSummary)
      }),
    )

    const user = userEvent.setup()
    render(<DescriptivePanel />)

    // AGE auto-loads on mount (slow), then immediately switch to GROUP (fast).
    await waitFor(() => expect(screen.getByText('AGE')).toBeInTheDocument())
    await user.click(screen.getByText('GROUP'))

    // GROUP's fast response should render...
    await waitFor(() =>
      expect(screen.getByText((_, el) => el?.textContent === 'Categorical · n=3')).toBeInTheDocument(),
    )
    // ...and AGE's slow response, arriving afterward, must NOT clobber it.
    await new Promise((r) => setTimeout(r, 80))
    expect(screen.getByText((_, el) => el?.textContent === 'Categorical · n=3')).toBeInTheDocument()
    expect(screen.queryByText((_, el) => el?.textContent === 'Continuous · n=3')).not.toBeInTheDocument()
  })
})
