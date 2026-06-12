#!/usr/bin/env python3
"""
scripts/anonymise.py
====================
Scrubs all tracked text and xlsx files for double-blind artifact submission.
Applies institutional and authorship de-identification per the replacement scheme
agreed in the plan:  C:/Users/pmar0042/.claude/plans/i-want-to-submit-linear-dolphin.md

Run from the repo root:
    python scripts/anonymise.py            # dry-run: report what would change
    python scripts/anonymise.py --apply    # apply changes in-place
    python scripts/anonymise.py --apply -v # verbose: print every changed file

This script is EXCLUDED from the artifact (do not include scripts/ in the zip).
"""

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── Replacement rules ─────────────────────────────────────────────────────────
# Applied in order (first match wins per position, but all rules sweep the
# whole string, so put most-specific patterns before general ones).
# Each entry: (regex_pattern_string, replacement_string).
# All patterns compiled without re.IGNORECASE unless prefixed with (?i).

RULES: list[tuple[str, str]] = [
    # ── Rater emails (most specific, before generic @uq rules) ───────────────
    (r'c\.wijenayake@uq\.edu\.au', 'rater1@example.edu'),
    (r't\.halloluwa@uq\.edu\.au', 'rater2@example.edu'),
    (r'u\.rathnayakemudiyanselage@uq\.edu\.au', 'rater3@example.edu'),

    # ── URLs (before bare domain rules) ──────────────────────────────────────
    (r'https?://study\.uq\.edu\.au', 'https://study.example.edu'),
    (r'https?://programs-courses\.uq\.edu\.au', 'https://programs-courses.example.edu'),
    (r'study\.uq\.edu\.au', 'study.example.edu'),
    (r'programs-courses\.uq\.edu\.au', 'programs-courses.example.edu'),
    # Any remaining @uq.edu.au address (local part may contain dots)
    (r'[A-Za-z0-9._%+\-]+@uq\.edu\.au', 'anon@example.edu'),
    (r'\buq\.edu\.au\b', 'example.edu'),

    # ── Rater bare local-parts and first names ────────────────────────────────
    (r'\bc\.wijenayake\b', 'rater1'),
    (r'\bt\.halloluwa\b', 'rater2'),
    (r'\bu\.rathnayakemudiyanselage\b', 'rater3'),
    # Surnames (distinctively associated with the three raters)
    (r'\bWijenayake\b', 'Rater1'),
    (r'\bHalloluwa\b', 'Rater2'),
    (r'\bRathnayakemudiyanselage\b', 'Rater3'),
    (r'\bRathnayake\b', 'Rater3'),
    (r'\bMudiyanselage\b', 'Rater3'),
    # First names (capitalized and lower)
    (r'\bChamith\b', 'Rater 1'),
    (r'\bchamith\b', 'rater1'),
    (r'\bThilina\b', 'Rater 2'),
    (r'\bthilina\b', 'rater2'),
    (r'\bUpul\b', 'Rater 3'),
    (r'\bupul\b', 'rater3'),

    # ── Author names ──────────────────────────────────────────────────────────
    (r'\bPasindu Marasinghe\b', 'Anonymous Author'),
    (r'\bPasindu\b', 'Anonymous'),
    (r'\bMarasinghe\b', 'Anonymous'),
    (r'\bmottretor\b', 'anon-user'),

    # ── Weights & Biases identifiers ─────────────────────────────────────────
    (r'\buq-unibot\b', 'anon-org'),
    (r'\buni-bot\b', 'anon-project'),
    (r'\bunibot\b', 'anonbot'),

    # ── Institutional name (longest/most specific first) ──────────────────────
    # Capture "The/the University of Queensland" → "the University"
    (r'[Tt]he University of Queensland', 'the University'),
    (r'University of Queensland', 'the University'),
    # Standalone UQ abbreviation (case-sensitive — "uq" in URLs already handled)
    (r"\bUQ's\b", "the University's"),
    (r'\bUQ\b', 'the University'),

    # ── Geographic / jurisdictional ───────────────────────────────────────────
    (r'\bBrisbane\b', 'the city'),
    # Also catch lowercase 'brisbane' inside compound domain names / URLs
    (r'brisbane', 'thecity'),
    (r'\bSt\.? Lucia\b', 'the main campus'),
    (r'\bQueensland\b', 'the state'),
    (r'\bQTAC\b', 'the state admissions centre'),
    (r'\bQCE\b', 'the senior secondary certificate'),

    # ── Article clean-up (must run AFTER institutional replacement) ───────────
    # "the UQ" → "the the University" → "the University"
    (r'\bthe the University\b', 'the University'),
    # "a UQ" → "a the University" → "a University"
    (r'\ba the University\b', 'a University'),
    (r'\ban the University\b', 'a University'),

    # ── Program codes → stable anonymous pseudo-codes ─────────────────────────
    # Order: longest ambiguous ones first; these are 4-digit numbers
    (r'\b2235\b', '9001'),   # BIT Honours
    (r'\b2453\b', '9002'),   # BIT (alternate code)
    (r'\b2555\b', '9003'),   # BIT / Design
    (r'\b2570\b', '9004'),   # BIT (main)
    (r'\b2571\b', '9005'),   # BusMgt / IT
    (r'\b2572\b', '9006'),   # Commerce / IT
    (r'\b2573\b', '9007'),   # HMNS / IT
    (r'\b2574\b', '9008'),   # IT / Arts
    (r'\b2575\b', '9009'),   # Eng(Hons) / IT
]

# Compile all patterns once at import time
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, flags=re.MULTILINE), repl) for pat, repl in RULES
]


def scrub(text: str) -> str:
    """Apply every replacement rule to *text* and return the result."""
    for pattern, repl in _COMPILED:
        text = pattern.sub(repl, text)
    return text


# ── File selectors ────────────────────────────────────────────────────────────

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    '.py', '.md', '.csv', '.jsonl', '.json', '.txt', '.sh', '.gitignore',
})

# Directories to always skip when walking for text files
SKIP_DIR_NAMES: frozenset[str] = frozenset({
    '.git', 'env', '__pycache__', 'graphify-out', 'scripts',
})


def _skip_path(p: pathlib.Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in p.parts)


def tracked_text_files() -> list[pathlib.Path]:
    """Return all git-tracked text files (by extension) outside skip dirs."""
    result = subprocess.run(
        ['git', 'ls-files'],
        capture_output=True, text=True, cwd=ROOT, check=True,
    )
    files: list[pathlib.Path] = []
    for line in result.stdout.splitlines():
        p = ROOT / line.strip()
        if p.suffix in TEXT_EXTENSIONS and not _skip_path(p):
            files.append(p)
    return files


def all_xlsx_files() -> list[pathlib.Path]:
    """Return all .xlsx files in the repo (excluding env/ and .git/)."""
    return [
        p for p in ROOT.rglob('*.xlsx')
        if not _skip_path(p)
    ]


# ── Processors ────────────────────────────────────────────────────────────────

def process_text(path: pathlib.Path, *, apply: bool, verbose: bool) -> bool:
    """Read, scrub, optionally write. Returns True if content changed."""
    try:
        original = path.read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        print(f'  [SKIP read error] {path.relative_to(ROOT)}: {exc}')
        return False

    scrubbed = scrub(original)
    if scrubbed == original:
        return False

    if verbose:
        print(f'  [CHANGED text] {path.relative_to(ROOT)}')
    if apply:
        path.write_text(scrubbed, encoding='utf-8')
    return True


def process_xlsx(path: pathlib.Path, *, apply: bool, verbose: bool) -> bool:
    """Scrub all string cell values and strip author metadata. Returns True if changed."""
    try:
        import openpyxl
        from openpyxl.utils.exceptions import InvalidFileException
    except ImportError:
        print('openpyxl not installed — XLSX files skipped. Run: pip install openpyxl')
        return False

    try:
        wb = openpyxl.load_workbook(path, keep_vba=False)
    except (InvalidFileException, Exception) as exc:
        print(f'  [SKIP xlsx error] {path.relative_to(ROOT)}: {exc}')
        return False

    changed = False

    # Scrub all cell values in every sheet (including hidden/very-hidden sheets)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    new_val = scrub(cell.value)
                    if new_val != cell.value:
                        cell.value = new_val
                        changed = True

    # Strip author metadata
    props = wb.properties
    for attr in ('creator', 'lastModifiedBy'):
        if getattr(props, attr, '') not in ('', 'Anonymous', None):
            setattr(props, attr, 'Anonymous')
            changed = True
    # Clear description / subject / keywords as well to be safe
    for attr in ('description', 'subject', 'keywords', 'category', 'contentStatus'):
        val = getattr(props, attr, None)
        if val and val.strip():
            setattr(props, attr, '')
            changed = True

    if changed:
        if verbose:
            print(f'  [CHANGED xlsx] {path.relative_to(ROOT)}')
        if apply:
            wb.save(path)

    return changed


# ── Renames ───────────────────────────────────────────────────────────────────

RATER_RENAMES: dict[str, str] = {
    'questionnaire_chamith.xlsx': 'questionnaire_rater1.xlsx',
    'questionnaire_thilina.xlsx': 'questionnaire_rater2.xlsx',
    'questionnaire_upul.xlsx':    'questionnaire_rater3.xlsx',
}

RETURNED_DIR = ROOT / 'study' / 'returned'


def rename_rater_files(*, apply: bool, verbose: bool) -> int:
    """Rename returned rater xlsx files. Returns number of renames done."""
    n = 0
    for old_name, new_name in RATER_RENAMES.items():
        old = RETURNED_DIR / old_name
        new = RETURNED_DIR / new_name
        if old.exists() and not new.exists():
            if verbose:
                print(f'  [RENAME] {old_name}  ->  {new_name}')
            if apply:
                old.rename(new)
            n += 1
        elif new.exists():
            if verbose:
                print(f'  [ALREADY RENAMED] {new_name}')
    return n


# ── PDF removal ───────────────────────────────────────────────────────────────

PDFS_TO_REMOVE: list[pathlib.Path] = [
    ROOT / 'data-collection' / 'sources' / 'international-guide-undergraduate-postgraduate.pdf',
    ROOT / 'data-collection' / 'sources' / 'domestic-guide-undergraduate.pdf',
]


def remove_pdfs(*, apply: bool, verbose: bool) -> int:
    """Delete the two branded PDF source documents. Returns number removed."""
    n = 0
    for pdf in PDFS_TO_REMOVE:
        if pdf.exists():
            if verbose:
                print(f'  [DELETE pdf] {pdf.relative_to(ROOT)}')
            if apply:
                pdf.unlink()
            n += 1
        else:
            if verbose:
                print(f'  [already gone] {pdf.name}')
    return n


# ── Verification ──────────────────────────────────────────────────────────────

VERIFY_PATTERNS = [
    r'[Qq]ueensland',
    r'\buq\.edu\.au\b',
    r'\bUQ\b',
    r'[Pp]asindu',
    r'[Mm]arasinghe',
    r'mottretor',
    r'[Cc]hamith',
    r'[Tt]hilina',
    r'[Uu]pul',
    r'[Ww]ijenayake',
    r'[Hh]alloluwa',
    r'[Rr]athnayake',
    r'uq-unibot',
    r'uni-bot',
    r'[Bb]risbane',
    r'St\.? Lucia',
    r'\bQTAC\b',
    r'\b(2235|2453|2555|2570|2571|2572|2573|2574|2575)\b',
]


def run_verification() -> bool:
    """Grep across tracked text files for residual identifying terms. Returns True if clean."""
    print('\n=== Verification grep ===')
    any_hit = False
    text_files = tracked_text_files()
    combined = re.compile('|'.join(VERIFY_PATTERNS))
    for p in text_files:
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        hits = [(i + 1, line) for i, line in enumerate(text.splitlines())
                if combined.search(line)]
        if hits:
            any_hit = True
            print(f'\n  {p.relative_to(ROOT)}:')
            for lineno, line in hits[:5]:
                print(f'    L{lineno}: {line.strip()[:120]}')
            if len(hits) > 5:
                print(f'    … and {len(hits) - 5} more lines')

    if any_hit:
        print('\n[WARN] Residual identifiers found - review the hits above.')
    else:
        print('[OK] No residual identifiers found in tracked text files.')
    return not any_hit


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Anonymise repo for double-blind artifact submission.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Write changes in-place (default: dry-run, no writes).',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print each changed file.',
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='After applying, run the verification grep sweep.',
    )
    args = parser.parse_args()

    if not args.apply:
        print('DRY-RUN — pass --apply to write changes.\n')

    total_changed = 0

    # 1. Text files
    print('=== (1/4) Text files ===')
    text_files = tracked_text_files()
    print(f'  Found {len(text_files)} tracked text files.')
    n = sum(
        process_text(p, apply=args.apply, verbose=args.verbose)
        for p in text_files
    )
    print(f'  {n} file(s) {"updated" if args.apply else "would change"}.')
    total_changed += n

    # 2. XLSX files
    print('\n=== (2/4) XLSX files ===')
    xlsx_files = all_xlsx_files()
    print(f'  Found {len(xlsx_files)} xlsx file(s).')
    n = sum(
        process_xlsx(p, apply=args.apply, verbose=args.verbose)
        for p in xlsx_files
    )
    print(f'  {n} file(s) {"updated" if args.apply else "would change"}.')
    total_changed += n

    # 3. Rename rater xlsx files
    print('\n=== (3/4) Rater file renames ===')
    n = rename_rater_files(apply=args.apply, verbose=True)
    print(f'  {n} rename(s) {"applied" if args.apply else "pending"}.')
    total_changed += n

    # 4. Remove branded PDFs
    print('\n=== (4/4) Remove branded PDFs ===')
    n = remove_pdfs(apply=args.apply, verbose=True)
    print(f'  {n} PDF(s) {"removed" if args.apply else "to remove"}.')
    total_changed += n

    print(f'\n{"[OK] Applied" if args.apply else "Dry-run complete"}: '
          f'{total_changed} total changes.')

    if args.apply and args.verify:
        run_verification()
    elif not args.apply:
        print('\nRe-run with --apply [--verify] to write changes.')


if __name__ == '__main__':
    main()
