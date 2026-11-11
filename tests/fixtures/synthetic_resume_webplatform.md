# Mara Voss

<!-- Synthetic résumé of a fictional persona (public-safe). Every person, employer, product,
client and number is invented. It is the ingest fixture: the 42 bullet sentences are byte-for-byte the bullet texts
of benchmarks_templates/synthetic_webplatform/vault_mara_voss.md, each job's "Technologies:" line
is the union of its projects' "Technologies used somewhere in this project" lines in
dev_flow/eval/t1/keyword_extraction_prompt.md, and the dates are the vault's From / To written as
prose. No real career data. -->

Toronto, ON · mara.voss@example.invalid · +1 416 555 0100 · github.com/mara-voss-example

## Summary

Web platform engineer with ten years across news publishing, e-commerce and education software.
Comfortable owning a service end to end — from the API and the data model to the front end,
the release pipeline and the on-call rotation — and used to working with editors, product and
support rather than only with other engineers.

## Experience

### Brindlewick Media — Senior Web Engineer
March 2021 – June 2025 · Toronto, ON (regional news publisher, 14 sites)

Technologies: Next.js, TypeScript, Node.js, Strapi, PostgreSQL, OAuth 2.0, OpenID Connect, Redis, REST, Webhooks, React, Storybook, CSS Modules, Elasticsearch, Python, FastAPI, Pydantic, LLM API, GitHub Actions, Docker, Terraform

**Brindlewick Headless CMS Migration**
- Led the move of 14 news sites from a monolithic PHP CMS to a headless setup with a Next.js front end, cutting median page weight by 38%.
- Brought Largest Contentful Paint on article pages from 4.1 s to 1.9 s through server-side rendering, edge image resizing and font subsetting.
- Built the content delivery API in TypeScript on Node.js with cursor pagination and per-route cache headers, serving 900 requests per second at peak.
- Ran the weekly migration review with editors and ad operations and kept the 14-site rollout calendar.

**Brindlewick Reader Accounts and Single Sign-On**
- Implemented single sign-on across 14 sites with an OpenID Connect provider, using the authorization-code flow with PKCE and rotating refresh tokens.
- Replaced long-lived cookies with short-lived access tokens and server-side session revocation, closing 3 findings from an external penetration test.
- Rebuilt the sign-in and registration forms to WCAG 2.1 AA and passed keyboard-only and screen-reader test rounds before release.

**Paywall and Subscription API**
- Integrated a third-party billing provider through REST calls and signed webhooks, with idempotency keys and a replay queue for failed deliveries.
- Designed the entitlement service that decides whether a reader may open an article, answering in under 20 ms at the 95th percentile.
- Added contract tests against the billing provider's sandbox that run on every merge and caught 6 breaking changes before release.

**Brindlewick Design System**
- Built a shared React component library of 46 components used by 5 product teams and published it as a versioned package.
- Added automated accessibility checks to every component and fixed 120 contrast and focus-order defects.
- Wrote usage guidelines and ran 4 workshops that moved 5 teams onto the library within two quarters.

**Editorial Search Rebuild**
- Rebuilt site search on Elasticsearch with per-language analyzers and synonym lists, raising click-through on the first result from 31% to 47%.
- Wrote the indexing pipeline that streams CMS changes into the search index within 5 seconds of publication.

**Newsroom Summaries Pilot**
- Built a pilot that drafts three-sentence article summaries with a hosted LLM API behind a Python FastAPI service, with editor review before publication.
- Scored 200 pilot summaries with editors against a five-point rubric and tracked the factual-error rate for each prompt version.
- Considered fine-tuning a small model for headlines and rejected it after the cost review; headlines stayed on the existing non-LLM template pipeline.

**Release Pipeline and Preview Environments**
- Built the CI/CD pipeline for 9 repositories with a preview environment per pull request, taking a merge to production from 45 minutes to 12.
- Described the staging and production environments in Terraform so that a new site could be stood up in a day instead of two weeks.
- Wrote a command-line scaffolder that creates a new service with linting, tests and deploy configuration in one command; used for 11 new services.

### Tallowmere Outfitters — Software Engineer, Storefront
February 2018 – February 2021 · Toronto, ON (outdoor-gear retailer)

Technologies: React, Node.js, REST, Docker, Webhooks, MySQL, RabbitMQ, Prometheus, Grafana, OpenTelemetry, Cypress, JavaScript, Sentry

**Tallowmere Storefront Replatform**
- Rebuilt the storefront as a server-rendered React application and cut time to interactive on product pages from 6.2 s to 2.8 s on mid-range phones.
- Compared a GraphQL gateway with the existing REST endpoints in a two-week spike and stayed with REST; wrote up the decision for the 3 storefront teams.
- Added an A/B testing layer to the storefront and ran 22 checkout experiments, the best of which raised conversion by 4%.

**Tallowmere Checkout and Payments Integration**
- Integrated 2 payment gateways and a tax-calculation service behind one checkout API, with retries and daily reconciliation reports.
- Added circuit breakers and timeouts around the gateways so that a provider outage degraded to one payment method instead of a failed checkout.
- Moved card entry to gateway-hosted fields so that card numbers never reached Tallowmere servers, which removed card storage from the compliance audit.

**Order Events Pipeline**
- Moved order processing from nightly batch files to RabbitMQ events consumed by 4 services, cutting the order-to-warehouse delay from hours to under a minute.
- Added dead-letter queues and a replay tool so that the on-call engineer could re-run a failed order without a developer.

**Storefront Monitoring and Alerting**
- Instrumented the storefront and checkout services with OpenTelemetry traces and built Grafana dashboards for latency, error rate and saturation.
- Rewrote the alert rules around service-level objectives and cut pages from 40 a week to 9.
- Set up the on-call rotation and the weekly incident review for 3 teams.

**End-to-End Test Suite for Checkout**
- Wrote 140 end-to-end checkout tests in Cypress that run on every pull request, replacing a two-day manual regression pass.
- Cut flaky test failures from 18% of runs to 2% by isolating test data and removing fixed waits.

**Release Readiness and Bug Triage**
- Set up weekly bug triage with support and product and brought the open-defect backlog from 310 to 85 in two quarters.
- Mentored 4 junior engineers through their first production releases with paired reviews and a release checklist.
- Traced a checkout crash affecting 3% of mobile sessions to a race in a third-party script and shipped a guarded loader.

### Quillhaven Learning — Full-Stack Engineer
May 2015 – December 2017 · Waterloo, ON (education software)

Technologies: Angular, TypeScript, Python, Django, PostgreSQL, REST

**Quillhaven Course Player**
- Built the lesson player used by 60,000 students, with offline progress sync for unreliable school networks.
- Added captions, transcript navigation and full keyboard control to the lesson player ahead of a district accessibility audit.

**Quillhaven Gradebook API**
- Built the gradebook REST API in Python and Django that 3 partner learning systems read through a published OpenAPI description.
- Migrated 8 years of grade records to a new PostgreSQL schema with a dual-write period and zero lost records.
- Onboarded 3 partner systems to the API with a sandbox, sample clients and a shared test checklist.

## Skills

Languages: TypeScript, JavaScript, Python, SQL
Front end: React, Next.js, Angular, accessibility (WCAG 2.1), design systems
Back end: Node.js, FastAPI, Django, REST API design, OAuth 2.0 / OpenID Connect, message queues
Operations: Docker, Terraform, GitHub Actions, OpenTelemetry, Grafana, on-call and incident review

## Education

B.Sc. Computer Science, Larkspur University, 2015
