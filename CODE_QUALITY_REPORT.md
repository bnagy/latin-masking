# Code Quality Report

## Summary

All **pyright** and **ruff** errors have been fixed. The codebase now passes:
- ✅ 0 pyright errors
- ✅ 0 ruff errors (only warnings remain)
- ✅ 82/82 tests passing

## Remaining Warnings (3277 total)

### By Category

| Count | Issue | Severity | Recommendation |
|-------|-------|----------|----------------|
| 125 | S101: Use of `assert` detected | Low | Acceptable in test files; consider pytest's assert rewriting |
| 10 | T201: `print` found | Low | Acceptable for CLI output |
| 2 | S310: Audit URL open for permitted schemes | Medium | Review URL handling for security |
| 2 | S301: `pickle` unsafe for untrusted data | Medium | Document that cache files are trusted |
| 1 | S108: Insecure temp directory | Low | Document that `/tmp` usage is intentional |
| 1 | B007: Loop control variable unused | Low | Remove or use the variable |

## Completed Fixes

### High Priority (Security/Complexity)

1. **B904: Exception chaining** - Fixed in `src/udpipe_masking/client.py`:
   - Added `from e` to all 4 exception raises in `_perform_request()`

2. **C901: Function complexity** - Refactored:
   - `parse_conllu()` in `src/udpipe_masking/conllu.py` - extracted `_process_frame()` helper
   - `main()` in `src/udpipe_masking/cli.py` - extracted `_cmd_*` subcommand handlers

### Medium Priority (Style/Readability)

3. **D413: Docstring formatting** - Fixed in:
   - `src/udpipe_masking/adverbs.py` (all functions)
   - `src/udpipe_masking/cache.py` (all functions)
   - `src/udpipe_masking/normalize.py` (all functions)
   - `src/udpipe_masking/client.py` (all functions)
   - `src/udpipe_masking/clitics.py` (all functions)

4. **D107: Missing `__init__` docstrings** - Fixed in `src/udpipe_masking/types.py`:
   - Added docstrings to `UDPipeError.__init__`
   - Added docstrings to `UDPipeAPIError.__init__`
   - Added docstrings to `UDPipeParseError.__init__`
   - Added docstrings to `UDPipeInputError.__init__`

### Low Priority (Auto-fixable)

5. **I001, E501** - Fixed by running `ruff format` and `ruff check --fix`

## Notes

- The `assert` statements (S101) are primarily in test files and are acceptable
- `print` statements (T201) are appropriate for CLI output
- `pickle` usage (S301) is acceptable for local cache files from trusted sources
- `/tmp` usage (S108) is intentional for cache bypass in regenerate mode