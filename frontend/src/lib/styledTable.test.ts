import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  buildStyledTableHtml,
  buildTsv,
  copyStyledTable,
  downloadStyledHtml,
  type StyledTableData,
} from './styledTable'

const sample: StyledTableData = {
  title: 'My Table',
  caption: 'A caption',
  columns: ['Name', 'Value'],
  rows: [
    ['a', '1'],
    ['b', '2'],
  ],
  filename: 'my-table',
}

describe('buildStyledTableHtml', () => {
  it('renders header cells for each column', () => {
    const html = buildStyledTableHtml(sample)
    expect(html).toContain('<th')
    expect(html).toContain('>Name<')
    expect(html).toContain('>Value<')
  })

  it('renders one <tr> per data row with matching cell values', () => {
    const html = buildStyledTableHtml(sample)
    expect(html).toContain('>a<')
    expect(html).toContain('>1<')
    expect(html).toContain('>b<')
    expect(html).toContain('>2<')
  })

  it('includes the title and caption when provided', () => {
    const html = buildStyledTableHtml(sample)
    expect(html).toContain('My Table')
    expect(html).toContain('A caption')
  })

  it('omits title/caption paragraphs when not provided', () => {
    const html = buildStyledTableHtml({ columns: ['A'], rows: [['1']] })
    expect(html).not.toContain('font-weight:bold;font-size:13px')
    expect(html).not.toContain('font-style:italic')
  })

  it('escapes HTML-special characters in cell content, title, and caption', () => {
    const data: StyledTableData = {
      title: '<script>alert(1)</script>',
      caption: 'A & B > C',
      columns: ['Col<1>'],
      rows: [['<b>bold</b> & "quoted"']],
    }
    const html = buildStyledTableHtml(data)
    expect(html).not.toContain('<script>')
    expect(html).toContain('&lt;script&gt;')
    expect(html).toContain('Col&lt;1&gt;')
    expect(html).toContain('&lt;b&gt;bold&lt;/b&gt; &amp; "quoted"')
    expect(html).toContain('A &amp; B &gt; C')
  })

  it('renders an empty string for missing cell values (row shorter than columns)', () => {
    const data: StyledTableData = { columns: ['A', 'B'], rows: [['only-a']] }
    const html = buildStyledTableHtml(data)
    // second <td> should be empty
    expect(html).toMatch(/<td[^>]*><\/td>/)
  })

  it('renders correctly with zero rows', () => {
    const data: StyledTableData = { columns: ['A', 'B'], rows: [] }
    const html = buildStyledTableHtml(data)
    expect(html).toContain('<thead>')
    expect(html).toContain('<tbody></tbody>')
  })

  it('renders correctly with zero columns', () => {
    const data: StyledTableData = { columns: [], rows: [[]] }
    const html = buildStyledTableHtml(data)
    expect(html).toContain('<tr></tr>')
  })

  it('wraps the table in a full HTML document when opts.fullDoc is true', () => {
    const html = buildStyledTableHtml(sample, { fullDoc: true })
    expect(html).toMatch(/^<!DOCTYPE html>/)
    expect(html).toContain('<html>')
    expect(html).toContain('<title>My Table</title>')
    expect(html).toContain('<body>')
  })

  it('uses a default document title of "Table" when no title given', () => {
    const html = buildStyledTableHtml({ columns: ['A'], rows: [['1']] }, { fullDoc: true })
    expect(html).toContain('<title>Table</title>')
  })

  it('handles unicode and emoji content without corruption', () => {
    const data: StyledTableData = { columns: ['Name'], rows: [['日本語 😀']] }
    const html = buildStyledTableHtml(data)
    expect(html).toContain('日本語 😀')
  })
})

describe('buildTsv', () => {
  it('joins the header with tabs', () => {
    const tsv = buildTsv(sample)
    expect(tsv.split('\n')[0]).toBe('Name\tValue')
  })

  it('joins each row with tabs and rows with newlines', () => {
    const tsv = buildTsv(sample)
    const lines = tsv.split('\n')
    expect(lines).toEqual(['Name\tValue', 'a\t1', 'b\t2'])
  })

  it('does not HTML-escape values (plain text target)', () => {
    const data: StyledTableData = { columns: ['A'], rows: [['<b>&x</b>']] }
    const tsv = buildTsv(data)
    expect(tsv).toContain('<b>&x</b>')
  })

  it('fills missing cells with empty string when a row is shorter than columns', () => {
    const data: StyledTableData = { columns: ['A', 'B'], rows: [['only-a']] }
    const tsv = buildTsv(data)
    expect(tsv.split('\n')[1]).toBe('only-a\t')
  })

  it('produces just the header line when there are no rows', () => {
    const data: StyledTableData = { columns: ['A', 'B'], rows: [] }
    const tsv = buildTsv(data)
    expect(tsv).toBe('A\tB\n')
  })
})

describe('copyStyledTable', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    // @ts-expect-error resetting test-only global
    delete (globalThis as { ClipboardItem?: unknown }).ClipboardItem
  })

  it('writes rich HTML + plain text via ClipboardItem when supported, and returns true', async () => {
    const writeSpy = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      value: { write: writeSpy, writeText: vi.fn() },
      configurable: true,
    })
    class FakeClipboardItem {
      constructor(public items: Record<string, Blob>) {}
    }
    // @ts-expect-error test-only global stub
    globalThis.ClipboardItem = FakeClipboardItem

    const ok = await copyStyledTable(sample)
    expect(ok).toBe(true)
    expect(writeSpy).toHaveBeenCalledTimes(1)
  })

  it('falls back to writeText(tsv) when ClipboardItem is unavailable', async () => {
    const writeTextSpy = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: writeTextSpy },
      configurable: true,
    })

    const ok = await copyStyledTable(sample)
    expect(ok).toBe(true)
    expect(writeTextSpy).toHaveBeenCalledWith(buildTsv(sample))
  })

  it('returns false when the clipboard write throws', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: {
        writeText: vi.fn().mockRejectedValue(new Error('denied')),
      },
      configurable: true,
    })

    const ok = await copyStyledTable(sample)
    expect(ok).toBe(false)
  })
})

describe('downloadStyledHtml', () => {
  let createObjectURL: ReturnType<typeof vi.fn>
  let revokeObjectURL: ReturnType<typeof vi.fn>
  let clickSpy: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.useFakeTimers()
    createObjectURL = vi.fn(() => 'blob:mock-url')
    revokeObjectURL = vi.fn()
    // @ts-expect-error jsdom stub
    URL.createObjectURL = createObjectURL
    // @ts-expect-error jsdom stub
    URL.revokeObjectURL = revokeObjectURL
    clickSpy = vi.fn()
    const origCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = origCreateElement(tag)
      if (tag === 'a') el.click = clickSpy
      return el
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('creates an object URL, triggers a click, and cleans up after a delay', () => {
    downloadStyledHtml(sample)
    expect(createObjectURL).toHaveBeenCalledTimes(1)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(revokeObjectURL).not.toHaveBeenCalled()
    vi.runAllTimers()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock-url')
  })

  it('uses the provided filename, defaulting to "table" when absent', () => {
    const appendSpy = vi.spyOn(document.body, 'appendChild')
    downloadStyledHtml({ columns: ['A'], rows: [['1']] })
    const anchor = appendSpy.mock.calls
      .map(([node]) => node as HTMLAnchorElement)
      .find((n) => n.tagName === 'A')
    expect(anchor?.download).toBe('table.html')
  })
})
