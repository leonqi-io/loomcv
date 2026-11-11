You are helping a person turn the bullets of their résumé into a searchable record.

For every bullet below, write the keywords a recruiter or a search system should be able to find
this bullet by. The person will review your list and correct it, so be accurate rather than generous.

Rules
1. A keyword is a short noun phrase: a skill, a technology, a method, a kind of system, or a
   business area. 3 to 8 keywords per bullet. Use the standard industry name of a thing.
2. Only what the person did. If the bullet says something was considered, compared, evaluated,
   rejected, avoided, or deliberately not done, that thing is NOT a keyword of the bullet.
3. Nothing that is not supported by the bullet or by its project's context line. No guesses about
   tools the text does not name.
4. Each project lists "technologies used somewhere in this project". Give one of those to a
   bullet only when the work that bullet describes was done with it. If you cannot tell, leave
   it out. A technology may belong to several bullets of a project, or to none.
5. When a bullet names only a specific instance of a well-known practice, you may add the
   common umbrella term for that practice as one extra keyword. At most one such term per bullet.
6. No soft skills (communication, teamwork), no seniority words, no numbers, no employer or
   product names.

Output
Return one JSON object and nothing else. Keys are the bullet ids exactly as given (P1b1, P1b2, …).
Each value is a list of keyword strings. Every bullet id must appear once.

Bullets
### Project 1: Brindlewick Headless CMS Migration
Context: Brindlewick Media (fictional regional news publisher), 2023–2024. Senior engineer leading the platform migration.
Technologies used somewhere in this project: Next.js; TypeScript; Node.js; Strapi; PostgreSQL
- P1b1: Led the move of 14 news sites from a monolithic PHP CMS to a headless setup with a Next.js front end, cutting median page weight by 38%.
- P1b2: Brought Largest Contentful Paint on article pages from 4.1 s to 1.9 s through server-side rendering, edge image resizing and font subsetting.
- P1b3: Built the content delivery API in TypeScript on Node.js with cursor pagination and per-route cache headers, serving 900 requests per second at peak.
- P1b4: Ran the weekly migration review with editors and ad operations and kept the 14-site rollout calendar.

### Project 2: Brindlewick Reader Accounts and Single Sign-On
Context: Brindlewick Media (fictional), 2022–2023. Engineer responsible for reader identity across all sites.
Technologies used somewhere in this project: OAuth 2.0; OpenID Connect; Node.js; Redis
- P2b1: Implemented single sign-on across 14 sites with an OpenID Connect provider, using the authorization-code flow with PKCE and rotating refresh tokens.
- P2b2: Replaced long-lived cookies with short-lived access tokens and server-side session revocation, closing 3 findings from an external penetration test.
- P2b3: Rebuilt the sign-in and registration forms to WCAG 2.1 AA and passed keyboard-only and screen-reader test rounds before release.

### Project 3: Paywall and Subscription API
Context: Brindlewick Media (fictional), 2024. Backend owner of the reader paywall.
Technologies used somewhere in this project: Node.js; TypeScript; REST; Webhooks; PostgreSQL
- P3b1: Integrated a third-party billing provider through REST calls and signed webhooks, with idempotency keys and a replay queue for failed deliveries.
- P3b2: Designed the entitlement service that decides whether a reader may open an article, answering in under 20 ms at the 95th percentile.
- P3b3: Added contract tests against the billing provider's sandbox that run on every merge and caught 6 breaking changes before release.

### Project 4: Brindlewick Design System
Context: Brindlewick Media (fictional), 2021–2022. Lead of the shared front-end library.
Technologies used somewhere in this project: React; TypeScript; Storybook; CSS Modules
- P4b1: Built a shared React component library of 46 components used by 5 product teams and published it as a versioned package.
- P4b2: Added automated accessibility checks to every component and fixed 120 contrast and focus-order defects.
- P4b3: Wrote usage guidelines and ran 4 workshops that moved 5 teams onto the library within two quarters.

### Project 5: Editorial Search Rebuild
Context: Brindlewick Media (fictional), 2023. Engineer on the site-search rebuild.
Technologies used somewhere in this project: Elasticsearch; Node.js; TypeScript
- P5b1: Rebuilt site search on Elasticsearch with per-language analyzers and synonym lists, raising click-through on the first result from 31% to 47%.
- P5b2: Wrote the indexing pipeline that streams CMS changes into the search index within 5 seconds of publication.

### Project 6: Newsroom Summaries Pilot
Context: Brindlewick Media (fictional), 2025. Engineer on a three-month newsroom pilot.
Technologies used somewhere in this project: Python; FastAPI; Pydantic; LLM API
- P6b1: Built a pilot that drafts three-sentence article summaries with a hosted LLM API behind a Python FastAPI service, with editor review before publication.
- P6b2: Scored 200 pilot summaries with editors against a five-point rubric and tracked the factual-error rate for each prompt version.
- P6b3: Considered fine-tuning a small model for headlines and rejected it after the cost review; headlines stayed on the existing non-LLM template pipeline.

### Project 7: Tallowmere Storefront Replatform
Context: Tallowmere Outfitters (fictional outdoor-gear retailer), 2019–2020. Front-end lead for the storefront.
Technologies used somewhere in this project: React; Node.js; REST; Docker
- P7b1: Rebuilt the storefront as a server-rendered React application and cut time to interactive on product pages from 6.2 s to 2.8 s on mid-range phones.
- P7b2: Compared a GraphQL gateway with the existing REST endpoints in a two-week spike and stayed with REST; wrote up the decision for the 3 storefront teams.
- P7b3: Added an A/B testing layer to the storefront and ran 22 checkout experiments, the best of which raised conversion by 4%.

### Project 8: Tallowmere Checkout and Payments Integration
Context: Tallowmere Outfitters (fictional), 2018–2019. Backend engineer on checkout.
Technologies used somewhere in this project: Node.js; REST; Webhooks; MySQL
- P8b1: Integrated 2 payment gateways and a tax-calculation service behind one checkout API, with retries and daily reconciliation reports.
- P8b2: Added circuit breakers and timeouts around the gateways so that a provider outage degraded to one payment method instead of a failed checkout.
- P8b3: Moved card entry to gateway-hosted fields so that card numbers never reached Tallowmere servers, which removed card storage from the compliance audit.

### Project 9: Order Events Pipeline
Context: Tallowmere Outfitters (fictional), 2020. Backend engineer on order processing.
Technologies used somewhere in this project: RabbitMQ; Node.js; Docker
- P9b1: Moved order processing from nightly batch files to RabbitMQ events consumed by 4 services, cutting the order-to-warehouse delay from hours to under a minute.
- P9b2: Added dead-letter queues and a replay tool so that the on-call engineer could re-run a failed order without a developer.

### Project 10: Storefront Monitoring and Alerting
Context: Tallowmere Outfitters (fictional), 2020–2021. Engineer responsible for storefront service health.
Technologies used somewhere in this project: Prometheus; Grafana; OpenTelemetry
- P10b1: Instrumented the storefront and checkout services with OpenTelemetry traces and built Grafana dashboards for latency, error rate and saturation.
- P10b2: Rewrote the alert rules around service-level objectives and cut pages from 40 a week to 9.
- P10b3: Set up the on-call rotation and the weekly incident review for 3 teams.

### Project 11: Release Pipeline and Preview Environments
Context: Brindlewick Media (fictional), 2022. Engineer on the platform tooling group.
Technologies used somewhere in this project: GitHub Actions; Docker; Terraform
- P11b1: Built the CI/CD pipeline for 9 repositories with a preview environment per pull request, taking a merge to production from 45 minutes to 12.
- P11b2: Described the staging and production environments in Terraform so that a new site could be stood up in a day instead of two weeks.
- P11b3: Wrote a command-line scaffolder that creates a new service with linting, tests and deploy configuration in one command; used for 11 new services.

### Project 12: End-to-End Test Suite for Checkout
Context: Tallowmere Outfitters (fictional), 2019. Engineer who started the automated regression suite.
Technologies used somewhere in this project: Cypress; JavaScript
- P12b1: Wrote 140 end-to-end checkout tests in Cypress that run on every pull request, replacing a two-day manual regression pass.
- P12b2: Cut flaky test failures from 18% of runs to 2% by isolating test data and removing fixed waits.

### Project 13: Quillhaven Course Player
Context: Quillhaven Learning (fictional education-software company), 2016–2017. Front-end engineer on the student product.
Technologies used somewhere in this project: Angular; TypeScript
- P13b1: Built the lesson player used by 60,000 students, with offline progress sync for unreliable school networks.
- P13b2: Added captions, transcript navigation and full keyboard control to the lesson player ahead of a district accessibility audit.

### Project 14: Quillhaven Gradebook API
Context: Quillhaven Learning (fictional), 2015–2016. Backend engineer on the gradebook.
Technologies used somewhere in this project: Python; Django; PostgreSQL; REST
- P14b1: Built the gradebook REST API in Python and Django that 3 partner learning systems read through a published OpenAPI description.
- P14b2: Migrated 8 years of grade records to a new PostgreSQL schema with a dual-write period and zero lost records.
- P14b3: Onboarded 3 partner systems to the API with a sandbox, sample clients and a shared test checklist.

### Project 15: Release Readiness and Bug Triage
Context: Tallowmere Outfitters (fictional), 2018. Engineer coordinating release quality for the web team.
Technologies used somewhere in this project: Sentry; JavaScript
- P15b1: Set up weekly bug triage with support and product and brought the open-defect backlog from 310 to 85 in two quarters.
- P15b2: Mentored 4 junior engineers through their first production releases with paired reviews and a release checklist.
- P15b3: Traced a checkout crash affecting 3% of mobile sessions to a race in a third-party script and shipped a guarded loader.
