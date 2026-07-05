import { describe, expect, it } from 'vitest'
import { fmtP, fmtPFull, fmtPubP, fmtPubPHtml, pCellTitle } from './format'

describe('fmtP', () => {
  it('returns em-dash for null/undefined', () => {
    expect(fmtP(null)).toBe('—')
    expect(fmtP(undefined)).toBe('—')
  })

  it('returns em-dash for NaN', () => {
    expect(fmtP(NaN)).toBe('—')
  })

  it('shows "<0.001" below the reporting floor, never "0.000"', () => {
    expect(fmtP(0.0009)).toBe('<0.001')
    expect(fmtP(0)).toBe('<0.001')
  })

  it('formats to 3 decimals otherwise', () => {
    expect(fmtP(0.035)).toBe('0.035')
    expect(fmtP(0.0431)).toBe('0.043')
    expect(fmtP(1)).toBe('1.000')
  })

  it('is the boundary-inclusive at exactly 0.001', () => {
    expect(fmtP(0.001)).toBe('0.001')
  })
})

describe('fmtPubP', () => {
  it('returns em-dash for null', () => {
    expect(fmtPubP(null)).toBe('—')
  })

  it('prefixes with p< below the floor', () => {
    expect(fmtPubP(0.0001)).toBe('p<0.001')
  })

  it('prefixes with p= otherwise', () => {
    expect(fmtPubP(0.035)).toBe('p=0.035')
  })
})

describe('fmtPFull', () => {
  it('returns em-dash for null', () => {
    expect(fmtPFull(null)).toBe('—')
  })

  it('uses scientific notation below 1e-4', () => {
    expect(fmtPFull(0.00001234)).toBe('1.234e-5')
  })

  it('trims trailing zeros for coarser values', () => {
    expect(fmtPFull(0.035)).toBe('0.035')
    expect(fmtPFull(0.5)).toBe('0.5')
  })
})

describe('pCellTitle', () => {
  it('renders a "p = —" placeholder for null', () => {
    expect(pCellTitle(null)).toBe('p = —')
  })

  it('renders full-precision value', () => {
    expect(pCellTitle(0.035)).toBe('p = 0.035')
  })
})

describe('fmtPubPHtml', () => {
  it('passes through the em-dash unmodified', () => {
    expect(fmtPubPHtml(null)).toBe('—')
  })

  it('wraps the leading "p" in <i> for Plotly text', () => {
    expect(fmtPubPHtml(0.035)).toBe('<i>p</i>=0.035')
    expect(fmtPubPHtml(0.0001)).toBe('<i>p</i><0.001')
  })
})
