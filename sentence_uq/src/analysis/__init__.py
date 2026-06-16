"""Read-only diagnostics over trained Bayesian sentence-level UQ models.

Modules here never modify training, inference, Fisher scoring, the feature
extractor, or the prior. They load already-trained artifacts and report
statistics that gate later modelling decisions (Phase 11-0 onward).
"""
