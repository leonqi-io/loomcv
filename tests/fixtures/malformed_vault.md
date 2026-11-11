# Malformed Synthetic Vault (fixture)

Fictional content exercising the strict-mode error paths. Each block below
carries exactly one deliberate defect.

## [Project] Old Grammar Project
* **Tags**: #fixture
* **Context**: Fictional project in the pre-R1 grammar.
* **Bullets**:
  - [Backend][API] A perfectly fine fictional bullet in a defective block.

## [Job] Missing From Job
* **To**: 2022
* **Bullets**:
  - [Backend][API] A fine fictional bullet in a job without a From line. {id: mf0001}

## [Job] Bad To Job
* **From**: 2020
* **To**: 2026-13
* **Bullets**:
  - A fine fictional bullet in a job whose To value is not a month. {id: bt0001}

## [Job] Missing Id Job
* **From**: 2019
* **To**: 2020
* **Bullets**:
  - [Frontend][UI] A valid fictional bullet with an id. {id: mi0001}
  - This bullet line has no id block and must be rejected.

## [Job] Empty Job
* **From**: 2018
* **To**: 2019
* **Bullets**:

## [Job] Twin Job
* **From**: 2016
* **To**: 2017
* **Bullets**:
  - [Tooling][CLI] Fictional bullet one. {id: tw0001}

## [Job] Twin Job
* **From**: 2014
* **To**: 2015
* **Bullets**:
  - [Tooling][CLI] Fictional bullet two. {id: tw0002}
