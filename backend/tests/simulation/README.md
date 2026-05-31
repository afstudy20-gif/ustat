# Phase 2 — Simulation & Property-Based Testing Infrastructure

Bu klasör, uSTAT'ı **orta-ileri düzey biyoistatistik uygulaması** haline getirmek için geliştirilen test altyapısının merkezidir.

## Amaç
- Bilinen gerçek parametrelerle üretilmiş verilerde yöntemlerin parametreleri ne kadar iyi geri getirdiğini doğrulamak.
- Hypothesis ile property-based testler yazarak çok çeşitli veri senaryolarında istatistiksel **invariant**'ları kontrol etmek.
- Basit endpoint smoke testlerinin yakalayamadığı istatistiksel regresyonları yakalamak.

## Nasıl Çalıştırılır

```bash
# Backend klasöründen
PYTHONPATH=. python -m pytest tests/simulation/ -q

# Sadece simulation testlerini çalıştırmak için
PYTHONPATH=. python -m pytest tests/simulation/ -q -m simulation
```

Hypothesis yüklü değilse ilgili testler otomatik olarak atlanır (şu anda manuel olarak `hypothesis` kurmanızı öneririz).

## Mevcut Yapı

| Dosya                              | Açıklama |
|------------------------------------|----------|
| `data_generators.py`               | Bilinen parametrelerle sentetik veri üretimi |
| `test_property_based.py`           | Hypothesis ile yazılmış gerçek property-based testler |
| `test_linear_simulation.py`        | Linear regression için klasik simulation recovery testleri |
| `test_logistic_simulation.py`      | Logistic regression için simulation testleri |
| `conftest.py`                      | Hypothesis stratejileri ve ortak ayarlar |

## Eklenen Property-Based Testler (Phase 2)

- `test_linear_recovers_sign_and_magnitude`: Farklı gürültü ve örnek sayılarında katsayıların işaret ve büyüklük açısından makul oranda geri kazanılması.
- `test_logistic_has_discrimination_power`: Gerçek sinyali olan verilerde AUC'un 0.5'in anlamlı üzerinde olması.
- `test_linear_does_not_crash_on_varied_data`: Çok çeşitli makul girdi kombinasyonlarında modelin çökmemesi (invariant).

## Yeni Test / Generator Eklerken

1. Gerekirse `data_generators.py`'ye yeni bir üretici fonksiyon ekle.
2. `test_xxx_simulation.py` veya `test_property_based.py` içine yeni `@given` testleri yaz.
3. Mümkünse hem klasik simulation hem de property-based versiyonunu yaz.
4. Testleri `tests/simulation/` altına koy.

## Gelecek Plan (Phase 2+)

- Survival (Cox) ve PSM/IPTW için güçlü simulation + property testleri
- Assumption violation altında davranış testleri (`services/assumptions.py` ile entegrasyon)
- Automated "recovery error" tabloları ve raporlar
- Daha karmaşık senaryolar (yüksek kollinearite, nadir olaylar, missing data mekanizmaları)

Bu altyapı, uSTAT'ın istatistiksel doğruluğunu orta-ileri seviyeye taşımanın en kritik adımlarından biridir.

### Phase 2 Strengthening (Latest)

- Added property-based tests for Survival (Cox) and PSM/IPTW.
- Added cross-phase test linking simulation with the new assumption checking system (Phase 1).
- Improved generators with better control over confounding strength and censoring.
- All new tests use Hypothesis strategies for broad coverage.
