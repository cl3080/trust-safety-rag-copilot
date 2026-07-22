# Red Flags for Fraudulent / Incentivized Reviews

Investigators and trust & safety teams commonly look for the following
signals when triaging suspected fake reviews. None is conclusive alone —
they are combined as weighted evidence.

## Linguistic signals
- Overuse of superlatives ("amazing," "perfect," "best ever") without
  specific, checkable details (room number, dish name, staff member).
- Heavy use of first-person singular pronouns ("I," "my") relative to
  genuine reviews, which more often reference third parties (front desk
  staff, other guests, specific amenities).
- Reviews that read like marketing copy — feature lists rather than a
  narrated experience.
- Low specificity: vague timeframes ("recently," "last month") instead of
  concrete dates or trip context.

## Behavioral / metadata signals
- Reviewer account created shortly before posting, with few or no other
  reviews.
- Burst posting: many 5-star reviews for the same listing within a short
  window, especially from accounts with little review history.
- Reviews posted from a narrow, unverified purchase/stay history.
- Duplicate or near-duplicate phrasing across multiple reviews (self-
  plagiarism), which suggests templated writing (human or LLM-generated).

## Structural signals
- Sentiment-review mismatch: a 5-star rating paired with lukewarm or
  neutral text, or vice versa.
- Reviews clustered right after a known incentive campaign (free product,
  discount code, "leave a review for a refund").

## Why this matters for LLM-assisted triage
A language model can be prompted to check a review against this list
explicitly (chain-of-thought grounded in policy) rather than emitting an
opaque "fake/real" label. That makes the verdict auditable, which matters
for any decision that could affect a seller's or reviewer's account.
