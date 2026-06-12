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
    (r'(?i)\buq\.edu\.au\b', 'example.edu'),
    # qtac.edu.au URL (before bare qtac rule)
    (r'(?i)qtac\.edu\.au', 'admissions.example.edu'),
    # lowercase 'uq' embedded in URL path slugs (e.g. find-approved-uq-agent)
    (r'(?i)-uq-', '-university-'),

    # ── Rater bare local-parts and first names ────────────────────────────────
    (r'\bc\.wijenayake\b', 'rater1'),
    (r'\bt\.halloluwa\b', 'rater2'),
    (r'\bu\.rathnayakemudiyanselage\b', 'rater3'),
    # Surnames (distinctively associated with the three raters)
    (r'(?i)\bWijenayake\b', 'Rater1'),
    (r'(?i)\bHalloluwa\b', 'Rater2'),
    (r'(?i)\bRathnayakemudiyanselage\b', 'Rater3'),
    (r'(?i)\bRathnayake\b', 'Rater3'),
    (r'(?i)\bMudiyanselage\b', 'Rater3'),
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

    # ── Institutional name (case-insensitive; longest/most specific first) ────
    (r'(?i)the University of Queensland', 'the University'),
    (r'(?i)University of Queensland', 'the University'),
    (r"(?i)\bUQ's\b", "the University's"),
    # Standalone UQ — case-insensitive so "uq" in any remaining slug is caught
    (r'(?i)\bUQ\b', 'the University'),

    # ── Geographic / jurisdictional (all case-insensitive) ────────────────────
    (r'(?i)\bBrisbane\b', 'the city'),
    # Catch 'brisbane' inside compound domain names / URL slugs
    (r'(?i)brisbane', 'thecity'),
    (r'(?i)St\.? Lucia\b', 'the main campus'),
    (r'(?i)\bQueensland\b', 'the state'),
    # Qld / QLD abbreviation (before bare QTAC rule)
    (r'(?i)\bQld\b', 'the state'),
    (r'(?i)\bQTAC\b', 'the state admissions centre'),
    (r'(?i)\bQCE\b', 'the senior secondary certificate'),

    # ── UQ campuses (other than St Lucia / main campus) ───────────────────────
    (r'(?i)\bGatton\b', 'the regional campus'),
    (r'(?i)\bHerston\b', 'another campus'),

    # ── UQ buildings / landmarks ──────────────────────────────────────────────
    (r'(?i)JD Story Building', 'the Administration Building'),
    (r'(?i)JD Story', 'the Administration Building'),
    (r'(?i)Forgan Smith\b', 'the main building'),
    (r'(?i)Sir Llew Edwards\b', 'a campus building'),
    (r'(?i)Sir Llew\b', 'a campus building'),

    # ── UQ Student Union acronym ──────────────────────────────────────────────
    (r'(?i)\bUQU\b', 'the Student Union'),

    # ── UQ-specific postcode ──────────────────────────────────────────────────
    (r'\b4072\b', '0000'),

    # NOTE: Grammar cleanup block moved to END of list so it runs after ALL
    # replacement rules (including Round 3 rules which can create new artifacts).

    # ── Program codes → stable anonymous pseudo-codes ─────────────────────────
    (r'\b2235\b', '9001'),   # BIT Honours
    (r'\b2453\b', '9002'),   # BIT (alternate code)
    (r'\b2555\b', '9003'),   # BIT / Design
    (r'\b2570\b', '9004'),   # BIT (main)
    (r'\b2571\b', '9005'),   # BusMgt / IT
    (r'\b2572\b', '9006'),   # Commerce / IT
    (r'\b2573\b', '9007'),   # HMNS / IT
    (r'\b2574\b', '9008'),   # IT / Arts
    (r'\b2575\b', '9009'),   # Eng(Hons) / IT

    # ── Round 3: CRICOS codes → stable pseudo-codes ───────────────────────────
    # Real CRICOS codes resolve directly to the exact institution + program via
    # the national registry.  Map each to a CRIC### pseudo that does NOT match
    # the \b\d{6}[A-Z]\b shape, so the verify sweep catches any stragglers.
    (r'\b001952K\b', 'CRIC001'),   # BIT (main)
    (r'\b027273G\b', 'CRIC002'),   # BIT / Arts
    (r'\b080731B\b', 'CRIC003'),   # BE(Hons) / IT
    (r'\b082962D\b', 'CRIC004'),   # BIT Honours
    (r'\b114806C\b', 'CRIC005'),   # BIT / Design
    (r'\b000011K\b', 'CRIC006'),
    (r'\b000194J\b', 'CRIC007'),
    (r'\b005911G\b', 'CRIC008'),
    (r'\b063374M\b', 'CRIC009'),
    (r'\b072417B\b', 'CRIC010'),
    (r'\b072856G\b', 'CRIC011'),
    (r'\b072877G\b', 'CRIC012'),
    (r'\b074021M\b', 'CRIC013'),
    (r'\b074522G\b', 'CRIC014'),
    (r'\b074601K\b', 'CRIC015'),
    (r'\b093862J\b', 'CRIC016'),
    (r'\b093863M\b', 'CRIC017'),
    (r'\b093912F\b', 'CRIC018'),

    # ── Round 3: Course codes → stable pseudo-prefixes (digits preserved) ──────
    # Discipline-specific prefixes are googleable to the institution.
    # Collapse onto neutral equivalents; keep the 4 digits so prereq chains
    # and any cross-file joins stay internally consistent.
    # CSSE (Computer Science/SE) → CORE  ((?i) so lowercase URL paths are caught)
    (r'(?i)\bCSSE(\d{4})\b', r'CORE\1'),
    # INFS (Information Systems) → INFO
    (r'(?i)\bINFS(\d{4})\b', r'INFO\1'),
    # DECO (Design Computing — very specific prefix) → DSGN
    (r'(?i)\bDECO(\d{4})\b', r'DSGN\1'),
    # COMP (appears in hallucinated model outputs for other unis) → CORE
    (r'(?i)\bCOMP(\d{4})\b', r'CORE\1'),
    # COMS → CORE
    (r'(?i)\bCOMS(\d{4})\b', r'CORE\1'),
    # CYBR / STAT → SUBJ (generic subject placeholder)
    (r'(?i)\bCYBR(\d{4})\b', r'SUBJ\1'),
    (r'(?i)\bSTAT(\d{4})\b', r'SUBJ\1'),
    # MATH and ENGG are universal prefixes used across many universities; keep.

    # ── Round 3: UQ student portal system (mySI-net) ──────────────────────────
    (r'(?i)\bmySI-net\b', 'the student portal'),
    (r'(?i)\bSI-net\b', 'the student portal'),

    # ── Round 3: UQ school / faculty names and acronyms ──────────────────────
    # Handle URL form first so the hostname is neutralised before the bare-token
    # rule runs (otherwise eecs.example.edu → "Computing.example.edu").
    (r'eecs\.example\.edu', 'cs.example.edu'),
    (r'(?i)School of Electrical Engineering and Computer Science', 'the School of Computing'),
    # Use 'Computing' (no leading 'the') to avoid "School of the School" artifact.
    (r'(?i)\bEECS\b', 'Computing'),
    # Handle compound EAIT phrases before the standalone acronym to avoid
    # "the Faculty faculty" and "the the Faculty" artifacts.
    (r'(?i)\bthe EAIT [Ff]aculty\b', 'the Faculty'),
    (r'(?i)\bEAIT [Ff]aculty\b', 'the Faculty'),
    (r'(?i)\bEAIT\b', 'the Faculty'),

    # ── Round 3: Named residential colleges / student housing ─────────────────
    (r'(?i)\bKev Carmody House\b', 'a new student residence'),
    (r'(?i)\bDuchesne\b', 'a college'),
    (r'(?i)\bCromwell\b', 'a college'),
    (r"(?i)\bSt Leo'?s\b", 'a college'),

    # ── Round 3: UQ career-development program brand ──────────────────────────
    # Handle "… Employability program" before bare phrase to avoid doubled word.
    (r'(?i)Enhance Your Employability\s+program\b', 'the career-development program'),
    (r'(?i)Enhance Your Employability', 'the career-development program'),

    # ── Grammar cleanup (MUST be last — runs after ALL replacement rules) ──────
    # Replacements earlier in this list can introduce "the the", "campus campus",
    # and similar artifacts.  These cleanup rules fix them in a final pass.
    # Capital-T sentence-initial "The the …" must precede the lowercase rule:
    (r'\bThe the\b', 'The'),
    # Mid-sentence: "the the …"
    (r'\bthe the\b', 'the'),
    # "a UQ …" → "a the University" → "a University"
    (r'\ba the University\b', 'a University'),
    (r'\ban the University\b', 'a University'),
    # "St Lucia campus" → "the main campus campus"
    (r'(?i)\bcampus campus\b', 'campus'),
    # Round-3 new artifacts:
    # "Enhance Your Employability program" → "the career-development program program"
    (r'(?i)\bprogram program\b', 'program'),
    # "the EAIT faculty" (when 'the' wasn't part of the matched phrase)
    (r'(?i)\bFaculty faculty\b', 'Faculty'),
    # "(EECS)" gloss after full-name replacement leaves "(Computing)"
    # after "School of Computing" → redundant but harmless; leave as-is.
    # "the Faculty the Faculty" (double replacement edge case)
    (r'(?i)\bthe Faculty the Faculty\b', 'the Faculty'),
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
# All patterns matched case-insensitively.
VERIFY_PATTERNS = [
    r'queensland',
    r'uq\.edu\.au',
    r'qtac',
    r'\bqld\b',
    r'gatton',
    r'herston',
    r'\b4072\b',
    r'\buqu\b',
    r'jd story',
    r'forgan smith',
    r'sir llew',
    r'pasindu',
    r'marasinghe',
    r'mottretor',
    r'chamith',
    r'thilina',
    r'\bupul\b',
    r'wijenayake',
    r'halloluwa',
    r'rathnayake',
    r'uq-unibot',
    r'uni-bot',
    r'brisbane',
    r'st\.? ?lucia',
    r'(?<![A-Za-z0-9])uq(?![A-Za-z0-9])',  # standalone uq any case
    r'\b(2235|2453|2555|2570|2571|2572|2573|2574|2575)\b',
    r'the the ',
    r'campus campus',
    # Round 3 additions
    r'\b\d{6}[A-Z]\b',                           # any remaining real CRICOS code shape
    r'\b(CSSE|INFS|DECO|COMP|COMS|CYBR|STAT)\d{4}\b',  # original course-code prefixes
    r'si-?net',                                   # mySI-net / SI-net
    r'\bEAIT\b',
    r'School of Electrical Engineering',
    r'Kev Carmody',
    r'Duchesne',
    r'Cromwell',
    r"St Leo'?s",
    r'Enhance Your Employability',
]

_VERIFY_COMBINED = re.compile(
    '|'.join(VERIFY_PATTERNS),
    flags=re.IGNORECASE | re.MULTILINE,
)


def _verify_text(text: str) -> list[tuple[int, str]]:
    """Return (lineno, line) pairs where _VERIFY_COMBINED matched."""
    return [
        (i + 1, line)
        for i, line in enumerate(text.splitlines())
        if _VERIFY_COMBINED.search(line)
    ]


def run_verification() -> bool:
    """Grep tracked text files AND xlsx XML internals for residual identifiers."""
    import zipfile as _zipfile
    any_hit = False

    print('\n=== Verification: text files ===')
    for p in tracked_text_files():
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        hits = _verify_text(text)
        if hits:
            any_hit = True
            print(f'\n  {p.relative_to(ROOT)}:')
            for lineno, line in hits[:5]:
                print(f'    L{lineno}: {line.strip()[:120]}')
            if len(hits) > 5:
                print(f'    ... and {len(hits) - 5} more lines')
    print('[OK] Text files clean.' if not any_hit else '')

    print('\n=== Verification: xlsx internals ===')
    xlsx_hit = False
    for xlsx in sorted(ROOT.rglob('*.xlsx')):
        if _skip_path(xlsx):
            continue
        file_hits: dict[str, list] = {}
        try:
            with _zipfile.ZipFile(xlsx) as zf:
                for member in zf.namelist():
                    if not member.endswith('.xml'):
                        continue
                    try:
                        content = zf.read(member).decode('utf-8', errors='replace')
                    except Exception:
                        continue
                    h = _verify_text(content)
                    if h:
                        file_hits[member] = h
        except Exception as exc:
            print(f'  [ERROR reading {xlsx.name}] {exc}')
            continue
        if file_hits:
            xlsx_hit = True
            print(f'\n  {xlsx.relative_to(ROOT)}:')
            for member, hits in file_hits.items():
                for lineno, line in hits[:3]:
                    print(f'    [{member}] L{lineno}: {line.strip()[:110]}')
    if xlsx_hit:
        any_hit = True
    print('[OK] XLSX internals clean.' if not xlsx_hit else '')

    print()
    if any_hit:
        print('[WARN] Overall: residual identifiers found - review above.')
    else:
        print('[OK] Overall: no residual identifiers found.')
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
