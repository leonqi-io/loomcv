# Status Block Fixture (synthetic)

All content is fictional and synthetic for parser exercise only.

## [Job] Malformed Block Unknown Key
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Valid bullet before the malformed one. {id: unk000}
    - [Frontend] Another bullet {id: unk001, new_key: value}
    - [API] Valid bullet after — parsing must continue. {id: unk002}

## [Job] Bad Status Value
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Valid bullet before the bad status. {id: bsv000}
    - [Frontend] Another bullet {id: bsv001, status: maybe}
    - [API] Valid bullet after — parsing must continue. {id: bsv002}

## [Job] Bad Date Format
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Valid bullet before the bad date. {id: bdf000}
    - [Frontend] Another bullet {id: bdf001, status: draft, updated: 2026-13-00}
    - [API] Valid bullet after — parsing must continue. {id: bdf002}

## [Job] Block Not At Line End
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Valid bullet before the misplaced block. {id: blk000}
    - [Frontend] Another {status: draft} bullet with text after the block. {id: blk002}
    - [API] Valid bullet after — parsing must continue. {id: blk003}

## [Job] Empty Text With Block
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Valid bullet before the empty-text bullet. {id: emp000}
    - [Frontend] {id: emp002, status: draft}
    - [API] Valid bullet after — parsing must continue. {id: emp003}

## [Job] Duplicate Key
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Valid bullet before the duplicate key. {id: dup000}
    - [Frontend] Another bullet {id: dup002, status: draft, status: confirmed}
    - [API] Valid bullet after — parsing must continue. {id: dup003}

## [Job] Whitespace Variants
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Compact form {id:ws0001,status:draft,updated:2026-08-20}
    - [Frontend] Spaced form { updated:2026-08-20 , id:ws0002 , status:draft }

## [Job] Date Only Block
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Date-only block {id: dat001, updated: 2026-08-20}

## [Job] Retired Leading Tags
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Valid bullet before retired tags. {id: ret001}
    - [user_confirmed] This uses a retired tag as a leading tag. {id: ret002}
    - [status] This uses a reserved tag name as a leading tag. {id: ret003}
    - [updated] This uses a reserved tag name as a leading tag. {id: ret004}
    - [2026-08-20] This uses a date-shaped token as a leading tag. {id: ret005}
    - [API] Valid bullet after — parsing must continue. {id: ret006}

## [Job] Provenance Prose Preserved
* **From**: 2020
* **To**: 2021
* **Bullets**:
    - [Backend] Text with provenance (CORRECTION, supersedes prior wording) {id: pro001, updated: 2026-08-20}
