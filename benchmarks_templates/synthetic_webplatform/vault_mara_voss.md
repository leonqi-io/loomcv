# Synthetic STAR Vault — Mara Voss (fictional persona, public-safe) — v3 job grammar

Every person, employer, product, client and number in this file is invented. It is the
second-persona dataset for evaluation and demos.
Each bullet's `Keywords` line is
`dev_flow/eval/t1/keywords_confirmed.json[<handle>]` verbatim; each `id` is the first 6 hex
characters of sha1 of the handle's `bullet_id`. No real career data.

Handle map (`P<n>b<m>` → `bullet_id`):

- P1b1 → brindlewick-media-senior-web-engineer::2b1f43
- P1b2 → brindlewick-media-senior-web-engineer::d134c6
- P1b3 → brindlewick-media-senior-web-engineer::8f398b
- P1b4 → brindlewick-media-senior-web-engineer::9ea8bb
- P2b1 → brindlewick-media-senior-web-engineer::f63573
- P2b2 → brindlewick-media-senior-web-engineer::bba446
- P2b3 → brindlewick-media-senior-web-engineer::9cca74
- P3b1 → brindlewick-media-senior-web-engineer::cd6a9d
- P3b2 → brindlewick-media-senior-web-engineer::4acca6
- P3b3 → brindlewick-media-senior-web-engineer::66f5ac
- P4b1 → brindlewick-media-senior-web-engineer::5ed7c7
- P4b2 → brindlewick-media-senior-web-engineer::b236ef
- P4b3 → brindlewick-media-senior-web-engineer::2bc7df
- P5b1 → brindlewick-media-senior-web-engineer::325730
- P5b2 → brindlewick-media-senior-web-engineer::cdd344
- P6b1 → brindlewick-media-senior-web-engineer::fbfec3
- P6b2 → brindlewick-media-senior-web-engineer::c5e922
- P6b3 → brindlewick-media-senior-web-engineer::59b4cd
- P11b1 → brindlewick-media-senior-web-engineer::8807cd
- P11b2 → brindlewick-media-senior-web-engineer::4e2f7b
- P11b3 → brindlewick-media-senior-web-engineer::8c3991
- P7b1 → tallowmere-outfitters-software-engineer-storefront::9e6d03
- P7b2 → tallowmere-outfitters-software-engineer-storefront::2c7e7b
- P7b3 → tallowmere-outfitters-software-engineer-storefront::35b69a
- P8b1 → tallowmere-outfitters-software-engineer-storefront::44a903
- P8b2 → tallowmere-outfitters-software-engineer-storefront::40984c
- P8b3 → tallowmere-outfitters-software-engineer-storefront::8ff484
- P9b1 → tallowmere-outfitters-software-engineer-storefront::2cd313
- P9b2 → tallowmere-outfitters-software-engineer-storefront::a92e0f
- P10b1 → tallowmere-outfitters-software-engineer-storefront::b33513
- P10b2 → tallowmere-outfitters-software-engineer-storefront::c6bc0f
- P10b3 → tallowmere-outfitters-software-engineer-storefront::166ad9
- P12b1 → tallowmere-outfitters-software-engineer-storefront::2f5a3b
- P12b2 → tallowmere-outfitters-software-engineer-storefront::6189dd
- P15b1 → tallowmere-outfitters-software-engineer-storefront::75b42f
- P15b2 → tallowmere-outfitters-software-engineer-storefront::f8b96c
- P15b3 → tallowmere-outfitters-software-engineer-storefront::7f81f3
- P13b1 → quillhaven-learning-full-stack-engineer::060761
- P13b2 → quillhaven-learning-full-stack-engineer::35cd3d
- P14b1 → quillhaven-learning-full-stack-engineer::bd9a5c
- P14b2 → quillhaven-learning-full-stack-engineer::de5007
- P14b3 → quillhaven-learning-full-stack-engineer::a2cc6f

---

## [Job] Brindlewick Media — Senior Web Engineer
* **Employer**: Brindlewick Media
* **Title**: Senior Web Engineer
* **From**: 2021-03
* **To**: 2025-06
* **Bullets**:
  - [Architecture][Migration] Led the move of 14 news sites from a monolithic PHP CMS to a headless setup with a Next.js front end, cutting median page weight by 38%. {id: 2b1f43, status: confirmed}
    * **Project**: Brindlewick Headless CMS Migration
    * **Keywords**: Headless CMS; CMS Migration; Next.js; Front-End Architecture; Web Performance Optimization; TypeScript; Node.js; Strapi; PostgreSQL
  - [Performance][Frontend] Brought Largest Contentful Paint on article pages from 4.1 s to 1.9 s through server-side rendering, edge image resizing and font subsetting. {id: d134c6, status: confirmed}
    * **Project**: Brindlewick Headless CMS Migration
    * **Keywords**: Web Performance Optimization; Core Web Vitals; Server-Side Rendering; Edge Computing; Image Optimization; Font Subsetting; Next.js; TypeScript; Node.js; Strapi; PostgreSQL
  - [Backend][API] Built the content delivery API in TypeScript on Node.js with cursor pagination and per-route cache headers, serving 900 requests per second at peak. {id: 8f398b, status: confirmed}
    * **Project**: Brindlewick Headless CMS Migration
    * **Keywords**: Content Delivery API; TypeScript; Node.js; API Development; Caching Headers; Database Pagination; Next.js; Strapi; PostgreSQL
  - [Planning][Stakeholders] Ran the weekly migration review with editors and ad operations and kept the 14-site rollout calendar. {id: 9ea8bb, status: confirmed}
    * **Project**: Brindlewick Headless CMS Migration
    * **Keywords**: Migration Planning; Rollout Management; Technical Program Management; Next.js; TypeScript; Node.js; Strapi; PostgreSQL
  - [Authentication][Backend] Implemented single sign-on across 14 sites with an OpenID Connect provider, using the authorization-code flow with PKCE and rotating refresh tokens. {id: f63573, status: confirmed}
    * **Project**: Brindlewick Reader Accounts and Single Sign-On
    * **Keywords**: Single Sign-On; OpenID Connect; OAuth 2.0; PKCE; Token Authentication; Identity Management; Node.js; Redis
  - [Security][Sessions] Replaced long-lived cookies with short-lived access tokens and server-side session revocation, closing 3 findings from an external penetration test. {id: bba446, status: confirmed}
    * **Project**: Brindlewick Reader Accounts and Single Sign-On
    * **Keywords**: Session Management; Access Tokens; Session Revocation; Vulnerability Remediation; Application Security; OAuth 2.0; OpenID Connect; Node.js; Redis
  - [Frontend][Accessibility] Rebuilt the sign-in and registration forms to WCAG 2.1 AA and passed keyboard-only and screen-reader test rounds before release. {id: 9cca74, status: confirmed}
    * **Project**: Brindlewick Reader Accounts and Single Sign-On
    * **Keywords**: Web Accessibility; WCAG; Form Design; Assistive Technology Testing; OAuth 2.0; OpenID Connect; Node.js; Redis
  - [Integration][Billing] Integrated a third-party billing provider through REST calls and signed webhooks, with idempotency keys and a replay queue for failed deliveries. {id: cd6a9d, status: confirmed}
    * **Project**: Paywall and Subscription API
    * **Keywords**: Payment Integration; REST APIs; Webhooks; Idempotency; Message Queues; Billing Systems; Node.js; TypeScript; PostgreSQL
  - [Backend][Latency] Designed the entitlement service that decides whether a reader may open an article, answering in under 20 ms at the 95th percentile. {id: 4acca6, status: confirmed}
    * **Project**: Paywall and Subscription API
    * **Keywords**: Entitlement Service; Authorization Systems; Paywall Architecture; Low-Latency Systems; Backend Architecture; Node.js; TypeScript; REST; Webhooks; PostgreSQL
  - [Testing][Contracts] Added contract tests against the billing provider's sandbox that run on every merge and caught 6 breaking changes before release. {id: 66f5ac, status: confirmed}
    * **Project**: Paywall and Subscription API
    * **Keywords**: Contract Testing; Integration Testing; Sandbox Testing; Continuous Integration; Node.js; TypeScript; REST; Webhooks; PostgreSQL
  - [Frontend][Components] Built a shared React component library of 46 components used by 5 product teams and published it as a versioned package. {id: 5ed7c7, status: confirmed}
    * **Project**: Brindlewick Design System
    * **Keywords**: Design Systems; React; Component Libraries; Front-End Development; Package Publishing; TypeScript; Storybook; CSS Modules
  - [Accessibility][Testing] Added automated accessibility checks to every component and fixed 120 contrast and focus-order defects. {id: b236ef, status: confirmed}
    * **Project**: Brindlewick Design System
    * **Keywords**: Automated Accessibility Testing; Web Accessibility; Design Systems; Defect Remediation; React; TypeScript; Storybook; CSS Modules
  - [Documentation][Adoption] Wrote usage guidelines and ran 4 workshops that moved 5 teams onto the library within two quarters. {id: 2bc7df, status: confirmed}
    * **Project**: Brindlewick Design System
    * **Keywords**: Technical Documentation; Design System Adoption; Developer Enablement; React; TypeScript; Storybook; CSS Modules
  - [Search][Backend] Rebuilt site search on Elasticsearch with per-language analyzers and synonym lists, raising click-through on the first result from 31% to 47%. {id: 325730, status: confirmed}
    * **Project**: Editorial Search Rebuild
    * **Keywords**: Search Engine Architecture; Elasticsearch; Site Search; Information Retrieval; Full-Text Search; Node.js; TypeScript
  - [Data][Pipeline] Wrote the indexing pipeline that streams CMS changes into the search index within 5 seconds of publication. {id: cdd344, status: confirmed}
    * **Project**: Editorial Search Rebuild
    * **Keywords**: Search Indexing; Data Pipelines; Event Streaming; Elasticsearch; Content Ingestion; Node.js; TypeScript
  - [AI][Prototype] Built a pilot that drafts three-sentence article summaries with a hosted LLM API behind a Python FastAPI service, with editor review before publication. {id: fbfec3, status: confirmed}
    * **Project**: Newsroom Summaries Pilot
    * **Keywords**: LLM Integration; FastAPI; Python; API Development; Prompt Engineering; Artificial Intelligence; Pydantic; LLM API
  - [Evaluation][Quality] Scored 200 pilot summaries with editors against a five-point rubric and tracked the factual-error rate for each prompt version. {id: c5e922, status: confirmed}
    * **Project**: Newsroom Summaries Pilot
    * **Keywords**: Model Evaluation; Prompt Evaluation; Error Analysis; Quality Assurance; Python; FastAPI; Pydantic; LLM API
  - [Architecture][Decision] Considered fine-tuning a small model for headlines and rejected it after the cost review; headlines stayed on the existing non-LLM template pipeline. {id: 59b4cd, status: confirmed}
    * **Project**: Newsroom Summaries Pilot
    * **Keywords**: Feasibility Analysis; Cost Analysis; Architecture Evaluation; Python; FastAPI; Pydantic; LLM API
  - [CICD][Tooling] Built the CI/CD pipeline for 9 repositories with a preview environment per pull request, taking a merge to production from 45 minutes to 12. {id: 8807cd, status: confirmed}
    * **Project**: Release Pipeline and Preview Environments
    * **Keywords**: CI/CD Pipelines; Preview Environments; Deployment Automation; DevOps; Build Automation; GitHub Actions; Docker; Terraform
  - [Infrastructure][Automation] Described the staging and production environments in Terraform so that a new site could be stood up in a day instead of two weeks. {id: 4e2f7b, status: confirmed}
    * **Project**: Release Pipeline and Preview Environments
    * **Keywords**: Infrastructure as Code; Terraform; Environment Provisioning; Cloud Infrastructure; GitHub Actions; Docker
  - [DeveloperExperience][Tooling] Wrote a command-line scaffolder that creates a new service with linting, tests and deploy configuration in one command; used for 11 new services. {id: 8c3991, status: confirmed}
    * **Project**: Release Pipeline and Preview Environments
    * **Keywords**: Scaffolding Tools; Developer Tooling; Command-Line Interfaces; Project Templates; GitHub Actions; Docker; Terraform

## [Job] Tallowmere Outfitters — Software Engineer, Storefront
* **Employer**: Tallowmere Outfitters
* **Title**: Software Engineer, Storefront
* **From**: 2018-02
* **To**: 2021-02
* **Bullets**:
  - [Frontend][Performance] Rebuilt the storefront as a server-rendered React application and cut time to interactive on product pages from 6.2 s to 2.8 s on mid-range phones. {id: 9e6d03, status: confirmed}
    * **Project**: Tallowmere Storefront Replatform
    * **Keywords**: Server-Side Rendering; React; E-Commerce Storefront; Front-End Architecture; Web Performance Optimization; Node.js; REST; Docker
  - [Architecture][Decision] Compared a GraphQL gateway with the existing REST endpoints in a two-week spike and stayed with REST; wrote up the decision for the 3 storefront teams. {id: 2c7e7b, status: confirmed}
    * **Project**: Tallowmere Storefront Replatform
    * **Keywords**: REST APIs; API Architecture; Technical Evaluation; Architecture Decision Records; React; Node.js; Docker
  - [Experimentation][Frontend] Added an A/B testing layer to the storefront and ran 22 checkout experiments, the best of which raised conversion by 4%. {id: 35b69a, status: confirmed}
    * **Project**: Tallowmere Storefront Replatform
    * **Keywords**: A/B Testing; Conversion Rate Optimization; Checkout Systems; Experimentation Platforms; React; Node.js; REST; Docker
  - [Integration][Payments] Integrated 2 payment gateways and a tax-calculation service behind one checkout API, with retries and daily reconciliation reports. {id: 44a903, status: confirmed}
    * **Project**: Tallowmere Checkout and Payments Integration
    * **Keywords**: Payment Gateway Integration; Checkout API; REST APIs; Tax Calculation; Financial Reconciliation; Node.js; Webhooks; MySQL
  - [Reliability][Backend] Added circuit breakers and timeouts around the gateways so that a provider outage degraded to one payment method instead of a failed checkout. {id: 40984c, status: confirmed}
    * **Project**: Tallowmere Checkout and Payments Integration
    * **Keywords**: Circuit Breaker Pattern; Fault Tolerance; System Resilience; Graceful Degradation; Payment Systems; Node.js; REST; Webhooks; MySQL
  - [Security][Compliance] Moved card entry to gateway-hosted fields so that card numbers never reached Tallowmere servers, which removed card storage from the compliance audit. {id: 8ff484, status: confirmed}
    * **Project**: Tallowmere Checkout and Payments Integration
    * **Keywords**: Payment Security; Hosted Fields; PCI Compliance; Security Compliance; Node.js; REST; Webhooks; MySQL
  - [Messaging][Backend] Moved order processing from nightly batch files to RabbitMQ events consumed by 4 services, cutting the order-to-warehouse delay from hours to under a minute. {id: 2cd313, status: confirmed}
    * **Project**: Order Events Pipeline
    * **Keywords**: Event-Driven Architecture; RabbitMQ; Order Processing; Message Brokers; Asynchronous Processing; Node.js; Docker
  - [Reliability][Operations] Added dead-letter queues and a replay tool so that the on-call engineer could re-run a failed order without a developer. {id: a92e0f, status: confirmed}
    * **Project**: Order Events Pipeline
    * **Keywords**: Dead-Letter Queues; Message Replay; Error Handling; Message Brokers; Site Reliability Engineering; RabbitMQ; Node.js; Docker
  - [Observability][Backend] Instrumented the storefront and checkout services with OpenTelemetry traces and built Grafana dashboards for latency, error rate and saturation. {id: b33513, status: confirmed}
    * **Project**: Storefront Monitoring and Alerting
    * **Keywords**: Distributed Tracing; OpenTelemetry; Grafana; Application Performance Monitoring; Observability Dashboards; Prometheus
  - [Alerting][Reliability] Rewrote the alert rules around service-level objectives and cut pages from 40 a week to 9. {id: c6bc0f, status: confirmed}
    * **Project**: Storefront Monitoring and Alerting
    * **Keywords**: Service-Level Objectives; Alerting Systems; Site Reliability Engineering; Alert Fatigue Reduction; Prometheus; Grafana; OpenTelemetry
  - [Process][OnCall] Set up the on-call rotation and the weekly incident review for 3 teams. {id: 166ad9, status: confirmed}
    * **Project**: Storefront Monitoring and Alerting
    * **Keywords**: Incident Management; On-Call Rotation; Post-Incident Review; Site Reliability Engineering; Prometheus; Grafana; OpenTelemetry
  - [Testing][Automation] Wrote 140 end-to-end checkout tests in Cypress that run on every pull request, replacing a two-day manual regression pass. {id: 2f5a3b, status: confirmed}
    * **Project**: End-to-End Test Suite for Checkout
    * **Keywords**: End-to-End Testing; Cypress; Test Automation; Regression Testing; Continuous Integration; JavaScript
  - [Quality][Flakiness] Cut flaky test failures from 18% of runs to 2% by isolating test data and removing fixed waits. {id: 6189dd, status: confirmed}
    * **Project**: End-to-End Test Suite for Checkout
    * **Keywords**: Test Flakiness Remediation; Test Isolation; Test Automation; Quality Engineering; Cypress; JavaScript
  - [Quality][Process] Set up weekly bug triage with support and product and brought the open-defect backlog from 310 to 85 in two quarters. {id: 75b42f, status: confirmed}
    * **Project**: Release Readiness and Bug Triage
    * **Keywords**: Bug Triage; Defect Management; Quality Assurance; Backlog Refinement; Sentry; JavaScript
  - [Leadership][Mentoring] Mentored 4 junior engineers through their first production releases with paired reviews and a release checklist. {id: f8b96c, status: confirmed}
    * **Project**: Release Readiness and Bug Triage
    * **Keywords**: Engineering Mentorship; Code Reviews; Release Management; Sentry; JavaScript
  - [Debugging][Frontend] Traced a checkout crash affecting 3% of mobile sessions to a race in a third-party script and shipped a guarded loader. {id: 7f81f3, status: confirmed}
    * **Project**: Release Readiness and Bug Triage
    * **Keywords**: Root Cause Analysis; Race Condition Debugging; Script Loading Optimization; Front-End Debugging; Sentry; JavaScript

## [Job] Quillhaven Learning — Full-Stack Engineer
* **Employer**: Quillhaven Learning
* **Title**: Full-Stack Engineer
* **From**: 2015-05
* **To**: 2017-12
* **Bullets**:
  - [Frontend][Product] Built the lesson player used by 60,000 students, with offline progress sync for unreliable school networks. {id: 060761, status: confirmed}
    * **Project**: Quillhaven Course Player
    * **Keywords**: Front-End Development; Offline Synchronization; Offline-First Architecture; Educational Technology; Angular; TypeScript
  - [Accessibility][Frontend] Added captions, transcript navigation and full keyboard control to the lesson player ahead of a district accessibility audit. {id: 35cd3d, status: confirmed}
    * **Project**: Quillhaven Course Player
    * **Keywords**: Web Accessibility; Closed Captioning; Keyboard Navigation; Accessibility Auditing; Angular; TypeScript
  - [Backend][API] Built the gradebook REST API in Python and Django that 3 partner learning systems read through a published OpenAPI description. {id: bd9a5c, status: confirmed}
    * **Project**: Quillhaven Gradebook API
    * **Keywords**: REST APIs; Python; Django; OpenAPI; API Design; PostgreSQL
  - [Data][Migration] Migrated 8 years of grade records to a new PostgreSQL schema with a dual-write period and zero lost records. {id: de5007, status: confirmed}
    * **Project**: Quillhaven Gradebook API
    * **Keywords**: Database Migration; PostgreSQL; Schema Migration; Dual-Write Pattern; Data Integrity; Python; Django; REST
  - [Integration][Partners] Onboarded 3 partner systems to the API with a sandbox, sample clients and a shared test checklist. {id: a2cc6f, status: confirmed}
    * **Project**: Quillhaven Gradebook API
    * **Keywords**: API Partner Onboarding; Developer Experience; API Sandbox; Integration Testing; Python; Django; PostgreSQL; REST
