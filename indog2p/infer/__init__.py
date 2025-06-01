# indog2p/infer/__init__.py

"""
indog2p.infer
=============

Entry point untuk pipeline inferensi IndoG2P:
- g2p_infer : pipeline G2P berbasis BERT (batch & single)
- Auto-load model, vocab, mapping dari config project
- Utilitas untuk deployment, REST API, dsb

Dapat diakses:
    from indog2p.infer import (
        g2p_infer,
        batch_g2p_predict,
        load_infer_pipeline
    )
"""

from .g2p_infer import (
    g2p_infer,
    batch_g2p_predict,
    load_infer_pipeline
)

__all__ = [
    "g2p_infer",
    "batch_g2p_predict",
    "load_infer_pipeline"
]
