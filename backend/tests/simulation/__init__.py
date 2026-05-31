"""
Simulation-based and Property-Based Testing Infrastructure (Phase 2)

Bu paket, uSTAT'ın istatistiksel yöntemlerini orta-ileri düzeyde doğrulamak için
geliştirilmiştir.

İçerik:
- `data_generators.py` → Bilinen parametrelerle sentetik veri üretimi
- `test_property_based.py` → Hypothesis ile property-based testler
- `test_*_simulation.py` → Klasik simulation recovery testleri
- `conftest.py` → Hypothesis stratejileri ve ayarlar

Kullanım:
    PYTHONPATH=. python -m pytest tests/simulation/ -q -m "simulation or not simulation"

Gelecek plan:
- Daha fazla model (Survival, PSM/IPTW, GEE)
- Automated "method vs ground truth" raporları
- Assumption violation altında davranış testleri
"""