# Regulatory and Marketplace Context for Review Fraud

## FTC rule on fake and manipulated reviews (finalized 2024)
The U.S. Federal Trade Commission adopted a rule that directly bans several
review-manipulation practices, including: buying or selling fabricated
consumer reviews, using reviews written by insiders without clear
disclosure, suppressing genuine negative reviews, and operating review
brokers that traffic in fake reviews or fake social-media engagement. The
rule allows the FTC to seek civil penalties per violation, which raised
the stakes for marketplaces to detect and remove manipulated reviews
proactively rather than reactively.

## Typical marketplace enforcement tiers
Most large marketplaces (Amazon, Yelp, TripAdvisor, app stores) use a
tiered response rather than a single fraud/not-fraud gate:
1. **Silent suppression** — a review is algorithmically down-weighted or
   hidden pending manual review; the account is not notified.
2. **Warning / education** — first-time low-confidence violations trigger
   a policy reminder rather than a penalty.
3. **Removal + account flag** — confirmed manipulation results in review
   deletion and an internal risk score bump on the account.
4. **Suspension** — repeated or high-confidence fraud (e.g., paid review
   rings) leads to account or seller suspension.

## Why retrieval-augmented generation fits this problem
Policy and enforcement thresholds change over time and vary by platform.
Hard-coding them into a classifier's training data goes stale. A RAG
approach keeps the policy text in a retrievable knowledge base that can
be updated independently of the model — the LLM is prompted to ground its
verdict in whatever policy snippets are retrieved for the current query,
rather than in frozen training-time knowledge.
