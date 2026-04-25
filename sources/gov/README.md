# `gov` — Federal & Government Standards

This collection covers U.S. federal cybersecurity frameworks, NIST standards, DOE/NNSA/DoD
directives, export control regulations, and nuclear security program rules — authoritative
sources for developers building software in regulated federal and national laboratory environments.

## What belongs here

### NIST Standards
- NIST SPs: SSDF (800-218), DevSecOps (800-204D), Zero Trust (800-207), Container Security (800-190)
- NIST AI frameworks: AI RMF (AI 100-1), Adversarial ML (AI 100-2), Synthetic Content (AI 100-4)
- NIST security control catalogs: 800-53r5, 800-53Ar5, 800-53B, 800-171r3, 800-172
- NIST systems engineering: 800-160v1r1/v2r1, CSF 2.0, RMF (800-37r2)
- NIST privacy/identity: 800-122 (PII), 800-188 (De-Identification), 800-63B (Digital Identity)

### DOE / NNSA / DoD Orders and Guides
- DOE Orders: 205.1B (Cyber Security), 206.1 (Privacy), 413.3B (Program Management),
  414.1D (Quality Assurance), 470.4B (Safeguards & Security), 472.2A (Personnel Security),
  Guide 414.1-4 (Safety Software), 10 CFR 830
- DoD DevSecOps: Container Hardening Guide v1.2
- Pre-processed markdown summaries (files 00–13) covering common DOE/NNSA software
  engineering requirements, security clearances, CUI handling, and RMF implementation

### Export Control Regulations (`ear/` and `itar/` subdirectories)
Converted from eCFR bulk XML — one markdown file per CFR Part, current as of download date.

- **EAR (15 CFR 730–774)** — Export Administration Regulations: general provisions, license
  requirements, license exceptions, country controls, end-user controls, embargo rules,
  enforcement, and the Commerce Control List (Part 774).
- **ITAR (22 CFR 120–130)** — International Traffic in Arms Regulations: definitions, USML
  categories, registration, licenses, exemptions, prohibited activities, violations,
  brokering, and political contributions.

### Human Reliability Program Regulations (`10cfr-hrp/` subdirectory)
Converted from eCFR bulk XML — current as of download date.

- **10 CFR 707** — Workplace Substance Abuse Programs at DOE Sites
- **10 CFR 710** — Procedures for Determining Eligibility for Access to Classified Matter
- **10 CFR 712** — Human Reliability Program (DOE/NNSA)

## Re-ingesting after updates

```bash
# Re-ingest the full gov collection (picks up all subdirectories automatically)
grounded-code-mcp ingest sources/gov --collection gov

# To refresh CFR content from updated eCFR XML:
# 1. Download fresh XML from https://www.ecfr.gov/current/title-XX (replace XX with title number)
# 2. Re-run scripts/ecfr_xml_to_md.py against the new XML
# 3. Re-ingest
```
