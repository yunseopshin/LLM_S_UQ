=== Pooling Diagnostic (setup 2, test split) ===
Content set: {ADJ, NOUN, NUM, PROPN}
Overall saturation fraction (tau=0.05): 0.42   [sanity vs known ~0.92]
Mean gate  content=0.0965  function=0.0969   ratio=1.0x
Saturation fraction  content=0.43  function=0.42
Epi_mask / Epi_unif:  median=1.30x  mean=1.62x  IQR=[0.99, 1.83]  (n=386)
Absolute Epi_mu:  unif median=0.00154   mask median=0.00187
mu shift (unif - mask):  median=-0.003   (function tokens inflate mu toward 1 when positive)
Sentences with |C_j|==0: 0.3%   |C_j|<=1: 0.8%

VERDICT: NO-GO. gate ratio 1.00x < 2x (content tokens are NOT clearly less saturated); median Epi lift 1.30x < 2x. Pooling will not rescue g_j; redirect effort to the loss/prior side (binomial focal, prior scale, or feature/layer changes). A uniform-saturation result is itself reportable.
