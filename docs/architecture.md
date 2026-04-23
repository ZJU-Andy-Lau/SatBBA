# Architecture Notes (Phase 1)

## Design Principles

1. **Stable boundaries**: config/model/matching/pipeline are isolated for long-term iteration.
2. **Plugin-first matching**: matchers are independently developed and swappable.
3. **Production-oriented infra**: typed config, logging, exceptions, tests, CLI.
4. **Deferred complexity**: BBA solver and triangulation are intentionally postponed.

## Module Responsibilities

- `config`: all runtime parameters and validation.
- `models`: canonical data contracts between modules.
- `matching`: pairwise feature matching abstraction and plugin registry.
- `io`: image/RPC cataloging and dataset loading.
- `core`: orchestration of end-to-end workflow.
- `logging`: centralized logger setup.
