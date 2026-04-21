# Programming Language Release History

A summary of major version releases for several widely-used programming languages, 2008–2024.

## Python

| Version | Release Date | Key Changes |
|---------|--------------|-------------|
| 3.0     | 2008-12-03   | Print as function, `str`/`bytes` split, removed old-style classes |
| 3.6     | 2016-12-23   | f-strings, ordered dicts, typing module improvements |
| 3.8     | 2019-10-14   | Walrus operator `:=`, positional-only parameters |
| 3.10    | 2021-10-04   | Structural pattern matching, better error messages |
| 3.11    | 2022-10-24   | ~25% speedup, `tomllib` in stdlib, exception groups |
| 3.12    | 2023-10-02   | Per-interpreter GIL work, improved f-string parsing |
| 3.13    | 2024-10-07   | Experimental free-threaded build (no-GIL), JIT prototype |

## Rust

| Version | Release Date | Key Changes |
|---------|--------------|-------------|
| 1.0     | 2015-05-15   | First stable release, stability guarantee |
| 1.31    | 2018-12-06   | Edition 2018, NLL borrow checker |
| 1.56    | 2021-10-21   | Edition 2021 |
| 1.70    | 2023-06-01   | `OnceCell`/`OnceLock`, sparse cargo registry |
| 1.80    | 2024-07-25   | LazyCell/LazyLock stabilized |

## JavaScript (ECMAScript)

| Version  | Year | Key Changes |
|----------|------|-------------|
| ES5      | 2009 | Strict mode, JSON, array methods |
| ES2015   | 2015 | let/const, classes, arrow functions, modules, promises |
| ES2017   | 2017 | async/await, Object.entries, shared memory |
| ES2020   | 2020 | BigInt, optional chaining, nullish coalescing |
| ES2022   | 2022 | Top-level await, class fields, error cause |
| ES2024   | 2024 | Object.groupBy, Promise.withResolvers |

## Go

| Version | Release Date | Key Changes |
|---------|--------------|-------------|
| 1.0     | 2012-03-28   | First stable release |
| 1.11    | 2018-08-24   | Go modules (experimental) |
| 1.18    | 2022-03-15   | Generics |
| 1.21    | 2023-08-08   | Built-in `min`/`max`/`clear`, `slog`, profile-guided optimization |
| 1.22    | 2024-02-06   | Range over integers, per-iteration loop variables |
| 1.23    | 2024-08-13   | Range-over-function iterators |

## Summary Statistics

- **Oldest surveyed language**: JavaScript (ES5 from 2009)
- **Most frequent releases**: Rust (roughly every 6 weeks; versions 1.0 through 1.80 span ~9 years)
- **Largest single version jump covered**: Python 2→3 via Python 3.0 in 2008
- **Languages introducing generics in this window**: Go (2022), Rust had them from 1.0

Release cadence varies dramatically. Rust and JavaScript publish on fixed schedules; Python follows an annual October cycle since 3.9; Go ships twice yearly in February and August. Major-version breakage is concentrated in two events: Python 2→3 (2008, painful 12-year migration) and ES5→ES2015 (2015, largely smooth due to transpilers like Babel).
