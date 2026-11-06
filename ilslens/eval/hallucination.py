"""Exact-match evidence without pretending token overlap is a hallucination judge."""
from .readout_tasks import contains_concept

def exact_readout_evidence(tokens,response):
    matched=[token for token in tokens if token.strip() and contains_concept(response,token.strip())]
    return dict(exact_matches=matched,claims=len(tokens),matched=len(matched),
                hallucination_rate=None,unjudged_claims=len(tokens),
                reason='A token bag has no asserted subject or polarity. Text absence or overlap cannot establish contradiction.')
