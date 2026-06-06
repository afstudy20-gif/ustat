"""Column-name suggestions: readable Sentence-case labels."""

from services.naming import suggest_display_name, suggest_names


def test_separators_to_sentence_case():
    assert suggest_display_name("blood_urea") == "Blood urea"
    assert suggest_display_name("ejection-fraction") == "Ejection fraction"


def test_camel_case_split():
    assert suggest_display_name("ejectionFraction") == "Ejection fraction"


def test_acronym_preserved():
    assert suggest_display_name("LDL CATEGORIES") == "LDL categories"
    assert suggest_display_name("DM") == "DM"


def test_all_caps_plain_word():
    assert suggest_display_name("SEX") == "Sex"


def test_digit_boundary():
    # 'ldl' is a known acronym, kept uppercase; digit split adds the space.
    assert suggest_display_name("ldl100") == "LDL 100"
    assert suggest_display_name("visit2") == "Visit 2"


def test_empty_and_blank():
    assert suggest_display_name("") == ""
    assert suggest_display_name("   ") == "   "


def test_suggest_names_omits_noops():
    out = suggest_names(["Age", "blood_urea", "DM"])
    # "Age" already sentence case -> omitted; "DM" acronym unchanged -> omitted.
    assert out == {"blood_urea": "Blood urea"}
