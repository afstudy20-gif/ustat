import { describe, expect, it, vi } from 'vitest'
import { applyResult, parseCommand, type ParseResult } from './commandParser'
import type { ColMeta } from '../store'

function col(name: string, kind: ColMeta['kind'], extra: Partial<ColMeta> = {}): ColMeta {
  return { name, dtype: kind === 'numeric' ? 'float64' : 'object', kind, ...extra }
}

const baseColumns: ColMeta[] = [
  col('age', 'numeric'),
  col('group', 'categorical'),
  col('sex', 'categorical'),
  col('score', 'numeric'),
  col('bmi', 'numeric'),
  col('outcome', 'categorical'),
]

describe('parseCommand', () => {
  describe('empty / whitespace input', () => {
    it('returns an empty result for an empty string', () => {
      const r = parseCommand('', baseColumns)
      expect(r.intent).toBeNull()
      expect(r.title).toBe('')
      expect(r.tab).toBe('')
      expect(r.fields).toEqual([])
      expect(r.complete).toBe(false)
      expect(r.preview).toBe('')
    })

    it('returns an empty result for whitespace-only input', () => {
      const r = parseCommand('   \t  ', baseColumns)
      expect(r.intent).toBeNull()
      expect(r.complete).toBe(false)
    })
  })

  describe('no intent recognised', () => {
    it('falls back to plain navigation when nothing matches', () => {
      const r = parseCommand('xyzzy plugh qux', baseColumns)
      expect(r.intent).toBeNull()
      expect(r.title).toBe('Search analyses…')
      expect(r.tab).toBe('')
      expect(r.fields).toEqual([])
      expect(r.complete).toBe(false)
      expect(r.preview).toBe('')
    })
  })

  describe('ROC intent', () => {
    it('recognises "roc" keyword and resolves both slots by kind', () => {
      const r = parseCommand('roc outcome vs score', baseColumns)
      expect(r.intent).toBe('roc')
      expect(r.title).toBe('ROC curve')
      expect(r.tab).toBe('roc')
      expect(r.complete).toBe(true)
      const outcome = r.fields.find((f) => f.key === 'outcomeCol')
      const score = r.fields.find((f) => f.key === 'scoreCol')
      expect(outcome?.value).toBe('outcome')
      expect(score?.value).toBe('score')
      expect(r.preview).toContain('ROC curve')
      expect(r.preview).toContain('Outcome=outcome')
      expect(r.preview).toContain('Score=score')
    })

    it('recognises the "roc curve" phrase', () => {
      const r = parseCommand('roc curve for group and age', baseColumns)
      expect(r.intent).toBe('roc')
    })

    it('recognises "auc" keyword', () => {
      const r = parseCommand('auc group age', baseColumns)
      expect(r.intent).toBe('roc')
    })

    it('is incomplete when a required slot is unfilled', () => {
      const r = parseCommand('roc age', baseColumns)
      expect(r.intent).toBe('roc')
      // "age" is numeric -> should fill scoreCol, outcomeCol left null
      const outcome = r.fields.find((f) => f.key === 'outcomeCol')
      expect(outcome?.value).toBeNull()
      expect(r.complete).toBe(false)
    })

    it('is incomplete with no columns supplied at all', () => {
      const r = parseCommand('roc', baseColumns)
      expect(r.intent).toBe('roc')
      expect(r.fields.every((f) => f.value === null)).toBe(true)
      expect(r.complete).toBe(false)
    })
  })

  describe('t-test intents', () => {
    it('recognises independent t-test phrase and fills col/groupCol by kind', () => {
      const r = parseCommand('independent t age by group', baseColumns)
      expect(r.intent).toBe('ttest_2sample')
      expect(r.tab).toBe('tests')
      const varField = r.fields.find((f) => f.key === 'col')
      const groupField = r.fields.find((f) => f.key === 'groupCol')
      expect(varField?.value).toBe('age')
      expect(groupField?.value).toBe('group')
      expect(r.complete).toBe(true)
    })

    it('recognises "ttest" word token', () => {
      const r = parseCommand('ttest age vs group', baseColumns)
      expect(r.intent).toBe('ttest_2sample')
    })

    it('recognises one-sample t phrase distinctly from two-sample', () => {
      const r = parseCommand('one sample t age', baseColumns)
      expect(r.intent).toBe('ttest_1sample')
      expect(r.fields).toHaveLength(1)
      expect(r.fields[0].value).toBe('age')
      expect(r.complete).toBe(true)
    })
  })

  describe('anova / mannwhitney / kruskal intents', () => {
    it('recognises anova keyword', () => {
      const r = parseCommand('anova bmi by group', baseColumns)
      expect(r.intent).toBe('anova')
      expect(r.complete).toBe(true)
    })

    it('recognises mann-whitney via hyphenated alias', () => {
      const r = parseCommand('mann-whitney score by sex', baseColumns)
      expect(r.intent).toBe('mannwhitney')
      expect(r.complete).toBe(true)
    })

    it('recognises "rank sum" phrase', () => {
      const r = parseCommand('rank sum score by sex', baseColumns)
      expect(r.intent).toBe('mannwhitney')
    })

    it('recognises kruskal-wallis phrase', () => {
      const r = parseCommand('kruskal-wallis bmi by group', baseColumns)
      expect(r.intent).toBe('kruskal')
      expect(r.complete).toBe(true)
    })

    it('recognises bare "kw" token', () => {
      const r = parseCommand('kw bmi by group', baseColumns)
      expect(r.intent).toBe('kruskal')
    })
  })

  describe('correlation intent', () => {
    it('recognises "correlation" and "pearson"/"spearman" aliases', () => {
      expect(parseCommand('correlation age and score', baseColumns).intent).toBe('correlation')
      expect(parseCommand('pearson age score', baseColumns).intent).toBe('correlation')
      expect(parseCommand('spearman age score', baseColumns).intent).toBe('correlation')
    })

    it('fills the single "vars" slot with the first matched numeric column', () => {
      const r = parseCommand('correlation age and score', baseColumns)
      expect(r.intent).toBe('correlation')
      expect(r.fields).toHaveLength(1)
      expect(r.fields[0].key).toBe('vars')
      expect(['age', 'score']).toContain(r.fields[0].value)
      expect(r.complete).toBe(true)
    })
  })

  describe('Turkish aliases and normalisation', () => {
    it('recognises "korelasyon" (Turkish for correlation)', () => {
      const r = parseCommand('korelasyon age score', baseColumns)
      expect(r.intent).toBe('correlation')
    })

    it('recognises "ki kare" style bağımsız t phrase with dotted i / Turkish chars', () => {
      const r = parseCommand('bağımsız t age group', baseColumns)
      expect(r.intent).toBe('ttest_2sample')
    })

    it('fuzzy-matches a column name with Turkish characters collapsed to ASCII', () => {
      const cols: ColMeta[] = [...baseColumns, col('yas', 'numeric', { label: 'yaş' })]
      const r = parseCommand('correlation yaş', cols)
      expect(r.intent).toBe('correlation')
      // "yaş" normalises to "yas" which matches the "yas" column exactly.
      expect(r.fields[0].value).toBe('yas')
    })
  })

  describe('malformed / noisy input', () => {
    it('ignores punctuation-heavy noise and still detects the intent', () => {
      const r = parseCommand('  ROC   ---   outcome ,,, vs   score!!  ', baseColumns)
      expect(r.intent).toBe('roc')
    })

    it('handles repeated/duplicate column tokens without duplicating fields', () => {
      const r = parseCommand('roc outcome vs outcome', baseColumns)
      expect(r.intent).toBe('roc')
      // Only one distinct matched candidate ("outcome"), so scoreCol stays null.
      const score = r.fields.find((f) => f.key === 'scoreCol')
      expect(score?.value).toBeNull()
    })

    it('does not crash on an empty column list', () => {
      const r = parseCommand('roc outcome vs score', [])
      expect(r.intent).toBe('roc')
      expect(r.fields.every((f) => f.value === null)).toBe(true)
      expect(r.complete).toBe(false)
    })

    it('does not match a column when nothing is close enough (Levenshtein > 2)', () => {
      const r = parseCommand('roc zzzzzzzz vs score', baseColumns)
      expect(r.intent).toBe('roc')
      const score = r.fields.find((f) => f.key === 'scoreCol')
      expect(score?.value).toBe('score')
      const outcome = r.fields.find((f) => f.key === 'outcomeCol')
      expect(outcome?.value).toBeNull()
    })

    it('handles quoted column names containing connector-like words', () => {
      const cols: ColMeta[] = [...baseColumns, col('time vs event', 'categorical')]
      const r = parseCommand('roc "time vs event" vs score', cols)
      expect(r.intent).toBe('roc')
      // The quoted phrase should be treated as a single token and matched,
      // rather than being split on the internal "vs" connector.
      const matchedNames = r.fields.map((f) => f.value);
      expect(matchedNames).toContain('time vs event')
    })

    it('matches via label/display_name when column name itself does not match', () => {
      const cols: ColMeta[] = [
        col('col_a', 'categorical', { label: 'outcome' }),
        col('col_b', 'numeric', { display_name: 'score' }),
      ]
      const r = parseCommand('roc outcome vs score', cols)
      expect(r.intent).toBe('roc')
      const outcome = r.fields.find((f) => f.key === 'outcomeCol')
      const score = r.fields.find((f) => f.key === 'scoreCol')
      expect(outcome?.value).toBe('col_a')
      expect(score?.value).toBe('col_b')
    })
  })

  describe('preview string', () => {
    it('omits the field list when no fields are filled', () => {
      const r = parseCommand('roc', baseColumns)
      expect(r.preview).toBe('ROC curve')
    })

    it('strips parenthetical hints from labels in the preview', () => {
      const r = parseCommand('roc outcome vs score', baseColumns)
      // Label is "Score (numeric)" -> preview key should be "Score", not "Score (numeric)".
      expect(r.preview).toContain('Score=score')
      expect(r.preview).not.toContain('(numeric)')
    })
  })
})

describe('applyResult', () => {
  it('navigates to the tab without touching panelCache when intent is null', () => {
    const setActiveTab = vi.fn()
    const setPanelCache = vi.fn()
    const result: ParseResult = {
      intent: null,
      title: 'Search analyses…',
      tab: 'roc',
      fields: [],
      complete: false,
      preview: '',
    }
    applyResult(result, { setActiveTab, setPanelCache })
    expect(setActiveTab).toHaveBeenCalledWith('roc')
    expect(setPanelCache).not.toHaveBeenCalled()
  })

  it('does nothing when intent is null and tab is empty', () => {
    const setActiveTab = vi.fn()
    const setPanelCache = vi.fn()
    applyResult(
      { intent: null, title: '', tab: '', fields: [], complete: false, preview: '' },
      { setActiveTab, setPanelCache },
    )
    expect(setActiveTab).not.toHaveBeenCalled()
    expect(setPanelCache).not.toHaveBeenCalled()
  })

  it('writes resolved fields into panelCache and switches tab for a simple intent (roc)', () => {
    const setActiveTab = vi.fn()
    const setPanelCache = vi.fn()
    const result = parseCommand('roc outcome vs score', baseColumns)
    applyResult(result, { setActiveTab, setPanelCache, panelCache: {} } as unknown as {
      setActiveTab: (t: string) => void
      setPanelCache: (panel: string, data: unknown) => void
    })
    expect(setPanelCache).toHaveBeenCalledWith(
      'roc',
      expect.objectContaining({ mode: 'single', outcomeCol: 'outcome', scoreCol: 'score' }),
    )
    expect(setActiveTab).toHaveBeenCalledWith('roc')
  })

  it('sets the combo sub-tab and testValue for a hypothesis-panel intent', () => {
    const setActiveTab = vi.fn()
    const setPanelCache = vi.fn()
    const result = parseCommand('ttest age by group', baseColumns)
    applyResult(
      result,
      { setActiveTab, setPanelCache, panelCache: {} } as unknown as {
        setActiveTab: (t: string) => void
        setPanelCache: (panel: string, data: unknown) => void
      },
    )
    expect(setPanelCache).toHaveBeenCalledWith('combo_tests', { sub: 'hypothesis' })
    expect(setPanelCache).toHaveBeenCalledWith(
      'hypothesis',
      expect.objectContaining({ test: 'ttest_2sample', col: 'age', groupCol: 'group' }),
    )
    expect(setActiveTab).toHaveBeenCalledWith('tests')
  })

  it('merges over existing panelCache instead of wiping unrelated fields', () => {
    const setActiveTab = vi.fn()
    const setPanelCache = vi.fn()
    const result = parseCommand('roc outcome vs score', baseColumns)
    applyResult(
      result,
      {
        setActiveTab,
        setPanelCache,
        panelCache: { roc: { unrelatedFlag: true } },
      } as unknown as { setActiveTab: (t: string) => void; setPanelCache: (panel: string, data: unknown) => void },
    )
    expect(setPanelCache).toHaveBeenCalledWith(
      'roc',
      expect.objectContaining({ unrelatedFlag: true, outcomeCol: 'outcome', scoreCol: 'score' }),
    )
  })

  it('accumulates correlation vars as a deduplicated string array', () => {
    const setActiveTab = vi.fn()
    const setPanelCache = vi.fn()
    const result = parseCommand('correlation age and score', baseColumns)
    applyResult(
      result,
      {
        setActiveTab,
        setPanelCache,
        panelCache: { correlation_pairwise: { vars: ['age'] } },
      } as unknown as { setActiveTab: (t: string) => void; setPanelCache: (panel: string, data: unknown) => void },
    )
    const call = setPanelCache.mock.calls.find(([panel]) => panel === 'correlation_pairwise')
    expect(call).toBeDefined()
    const vars = (call?.[1] as { vars: string[] }).vars
    expect(new Set(vars).size).toBe(vars.length)
    expect(vars).toContain('age')
  })

  it('does nothing when the schema for the intent cannot be found', () => {
    const setActiveTab = vi.fn()
    const setPanelCache = vi.fn()
    applyResult(
      {
        intent: 'not_a_real_intent',
        title: '',
        tab: 'somewhere',
        fields: [],
        complete: false,
        preview: '',
      },
      { setActiveTab, setPanelCache },
    )
    expect(setActiveTab).not.toHaveBeenCalled()
    expect(setPanelCache).not.toHaveBeenCalled()
  })
})
