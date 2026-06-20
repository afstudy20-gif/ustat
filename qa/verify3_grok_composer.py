"""Quick extras: frequency on sex, descriptive all-cols, histogram on text col."""
import sys
sys.path.insert(0, "qa")
from run_via_testclient import boot
client, sid = boot()


def safe_get(path, **p):
    try:
        return client.get(path, params=p)
    except Exception as exc:
        class _E:
            status_code = 599; text = str(exc); content = b""
            def json(self): return {"_exc": str(exc)}
        return _E()


def safe_post(path, body):
    try:
        return client.post(path, json=body)
    except Exception as exc:
        class _E:
            status_code = 599; text = str(exc); content = b""
            def json(self): return {"_exc": str(exc)}
        return _E()


# Frequency of sex — does it surface "x"/"Female"/blank as separate cats?
print("=== frequency(sex) ===")
r = safe_get(f"/api/stats/{sid}/frequency", column="sex")
b = r.json()
print("  status=", r.status_code)
for cat in b.get("sex", {}).get("categories", []):
    print("   ", cat)

# Histogram on bmi (text) — crash or coerce?
print("\n=== histogram on bmi (text) ===")
r = safe_post("/api/charts/histogram", {"session_id": sid, "x": "bmi", "bins": 10})
print("  status=", r.status_code)
if r.status_code < 599:
    try:
        print("  body keys:", list(r.json().keys()))
    except Exception:
        print("  non-json")
else:
    print("  exc:", r.text[:200])

# Subgroup bar where y_col is text (bmi) in mean mode
print("\n=== subgroup_bar mean on text y_col=bmi ===")
r = safe_post("/api/charts/subgroup_bar", {
    "session_id": sid, "y_col": "bmi", "subgroup_col": "diabetes",
    "xaxis_col": "nyha", "y_mode": "mean",
})
print("  status=", r.status_code)
if r.status_code < 599:
    for tr in r.json().get("traces", []):
        print("   ", tr["name"], tr["y"], tr["ns"])
else:
    print("  exc:", r.text[:200])

# Table1 with NO group_column (overall only) — does it still work?
print("\n=== table1 no group ===")
r = client.post("/api/stats/table1", json={
    "session_id": sid, "variables": ["age", "nyha"],
})
b = r.json()
print("  status=", r.status_code, "group_labels=", b.get("group_labels"),
      "total_n=", b.get("total_n"))
for row in b.get("rows", []):
    print(f"   {row['variable']}: type={row['type']} overall={row.get('overall')}")

# descriptive with no column (all numeric)
print("\n=== descriptive all numeric ===")
r = safe_get(f"/api/stats/{sid}/descriptive")
b = r.json()
print("  status=", r.status_code, "cols=", list(b.keys()) if isinstance(b, dict) else type(b))

print("\nDONE3")
