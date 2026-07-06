import { describe, expect, it } from 'vitest'
import { INTENT_SCHEMAS, SCHEMA_BY_INTENT, slotAccepts } from './commandSchema'

describe('slotAccepts', () => {
  it('accepts any kind when slot is "any"', () => {
    expect(slotAccepts('any', 'numeric')).toBe(true)
    expect(slotAccepts('any', 'categorical')).toBe(true)
    expect(slotAccepts('any', 'ordinal')).toBe(true)
    expect(slotAccepts('any', 'text')).toBe(true)
    expect(slotAccepts('any', 'date')).toBe(true)
  })

  it('numeric slot accepts numeric and ordinal only', () => {
    expect(slotAccepts('numeric', 'numeric')).toBe(true)
    expect(slotAccepts('numeric', 'ordinal')).toBe(true)
    expect(slotAccepts('numeric', 'categorical')).toBe(false)
    expect(slotAccepts('numeric', 'text')).toBe(false)
    expect(slotAccepts('numeric', 'date')).toBe(false)
  })

  it('categorical slot accepts categorical and ordinal only', () => {
    expect(slotAccepts('categorical', 'categorical')).toBe(true)
    expect(slotAccepts('categorical', 'ordinal')).toBe(true)
    expect(slotAccepts('categorical', 'numeric')).toBe(false)
    expect(slotAccepts('categorical', 'text')).toBe(false)
    expect(slotAccepts('categorical', 'date')).toBe(false)
  })
})

describe('INTENT_SCHEMAS', () => {
  it('is a non-empty array of schemas with required base fields', () => {
    expect(INTENT_SCHEMAS.length).toBeGreaterThan(0)
    for (const s of INTENT_SCHEMAS) {
      expect(typeof s.intent).toBe('string')
      expect(s.intent.length).toBeGreaterThan(0)
      expect(typeof s.title).toBe('string')
      expect(typeof s.tab).toBe('string')
      expect(typeof s.panelId).toBe('string')
      expect(Array.isArray(s.fields)).toBe(true)
    }
  })

  it('has unique intent ids', () => {
    const ids = INTENT_SCHEMAS.map((s) => s.intent)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('includes the documented MVP intents', () => {
    const ids = INTENT_SCHEMAS.map((s) => s.intent)
    expect(ids).toEqual(
      expect.arrayContaining([
        'roc',
        'ttest_2sample',
        'ttest_1sample',
        'anova',
        'mannwhitney',
        'kruskal',
        'correlation',
      ]),
    )
  })

  it('roc schema declares fixed mode and two required fields', () => {
    const roc = INTENT_SCHEMAS.find((s) => s.intent === 'roc')
    expect(roc).toBeDefined()
    expect(roc?.fixed).toEqual({ mode: 'single' })
    expect(roc?.fields).toHaveLength(2)
    expect(roc?.fields.every((f) => f.required)).toBe(true)
  })

  it('hypothesis-backed intents declare comboId/comboSub/testValue', () => {
    const hypothesisIntents = ['ttest_2sample', 'ttest_1sample', 'anova', 'mannwhitney', 'kruskal']
    for (const intent of hypothesisIntents) {
      const schema = INTENT_SCHEMAS.find((s) => s.intent === intent);
      expect(schema?.comboId).toBe('combo_tests')
      expect(schema?.comboSub).toBe('hypothesis')
      expect(schema?.testValue).toBe(intent)
      expect(schema?.panelId).toBe('hypothesis')
    }
  })

  it('correlation schema only requires a numeric "vars" slot', () => {
    const corr = INTENT_SCHEMAS.find((s) => s.intent === 'correlation')
    expect(corr?.fields).toHaveLength(1)
    expect(corr?.fields[0]).toMatchObject({ key: 'vars', kind: 'numeric', required: true })
  })

  it('ttest_1sample has exactly one required numeric field (no groupCol)', () => {
    const one = INTENT_SCHEMAS.find((s) => s.intent === 'ttest_1sample')
    expect(one?.fields).toHaveLength(1)
    expect(one?.fields[0].key).toBe('col')
  })
})

describe('SCHEMA_BY_INTENT', () => {
  it('maps every intent id from INTENT_SCHEMAS to its schema', () => {
    for (const s of INTENT_SCHEMAS) {
      expect(SCHEMA_BY_INTENT[s.intent]).toBe(s)
    }
  })

  it('returns undefined for unknown intents', () => {
    expect(SCHEMA_BY_INTENT['not_a_real_intent']).toBeUndefined()
  })

  it('has exactly as many keys as INTENT_SCHEMAS entries', () => {
    expect(Object.keys(SCHEMA_BY_INTENT)).toHaveLength(INTENT_SCHEMAS.length)
  })
})
