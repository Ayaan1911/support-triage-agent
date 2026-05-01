"""
main.py — CLI entry point for the Multi-Domain Support Triage Agent.

Usage:
    python main.py --crawl                          # Crawl support sites → corpus/
    python main.py --crawl --force                  # Force re-crawl even if corpus exists
    python main.py --build-index                    # Build TF-IDF index from corpus/
    python main.py --run                            # Process data/support_tickets.csv → output.csv
    python main.py --run --input data/sample_support_tickets.csv
    python main.py --all                            # crawl + build-index + run
    python main.py --all --force                    # Force re-crawl then run
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (relative to this script's location)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / ".." / "data"
DATA_DIR = BASE_DIR / ".." / "data"
DEFAULT_INPUT = BASE_DIR / ".." / "support_issues" / "support_issues.csv"
OUTPUT_CSV = BASE_DIR / ".." / "support_issues" / "output.csv"
LOG_FILE = BASE_DIR / "log.txt"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Configure root logger to write INFO+ to stdout and DEBUG+ to log.txt."""
    import io

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Wrap stdout in a UTF-8 text stream to avoid Windows cp1252 encoding errors
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    stream_handler = logging.StreamHandler(utf8_stdout)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[stream_handler, file_handler],
    )
    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "requests", "httpcore", "httpx", "groq"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Log-ticket writer
# ---------------------------------------------------------------------------


def _write_ticket_log(
    ticket_num: int,
    issue: str,
    company: str,
    decision: str,
    retrieved_docs: list,
    response: str,
    justification: str,
) -> None:
    """Append a formatted ticket entry to log.txt."""
    doc_list = retrieved_docs if retrieved_docs else ["(none)"]
    doc_str = ", ".join(os.path.basename(d) for d in doc_list)

    entry = (
        f"\n=== TICKET #{ticket_num} ===\n"
        f"ISSUE: {issue}\n"
        f"COMPANY: {company}\n"
        f"DECISION: {decision}\n"
        f"RETRIEVED DOCS: [{doc_str}]\n"
        f"RESPONSE: {response}\n"
        f"JUSTIFICATION: {justification}\n"
    )

    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(entry)


# ---------------------------------------------------------------------------
# Step 1: --crawl
# ---------------------------------------------------------------------------


def cmd_crawl(force: bool) -> None:
    """Run the web crawler for all configured domains."""
    from crawler import run_crawler

    print(f"\n{'='*60}")
    print("  STEP 1: CRAWLING SUPPORT DOCUMENTATION")
    print(f"{'='*60}\n")

    run_crawler(corpus_dir=CORPUS_DIR, force=force)


# ---------------------------------------------------------------------------
# Step 2: --build-index
# ---------------------------------------------------------------------------


def cmd_build_index() -> "Retriever":  # type: ignore[name-defined]
    """Build and return the TF-IDF retriever index."""
    from retriever import Retriever

    print(f"\n{'='*60}")
    print("  STEP 2: BUILDING TF-IDF INDEX")
    print(f"{'='*60}\n")

    retriever = Retriever(str(CORPUS_DIR))

    if retriever.corpus_is_empty():
        print(
            "[main] WARNING: Corpus is empty. Run --crawl first to populate it.\n"
            "       The agent will still run but will have no documentation to retrieve from."
        )
    else:
        retriever.build_index()
        print("[main] Index built successfully.\n")

    return retriever


# ---------------------------------------------------------------------------
# Step 3: --run
# ---------------------------------------------------------------------------


def cmd_run(input_path: Path, retriever) -> None:
    """Process all tickets and write results to output.csv."""
    from agent import TriageAgent

    print(f"\n{'='*60}")
    print("  STEP 3: PROCESSING TICKETS")
    print(f"{'='*60}\n")

    # Load input CSV
    if not input_path.exists():
        print(f"[main] ERROR: Input file not found: {input_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="latin-1")

    # Validate required columns
    required_cols = {"Issue", "Subject", "Company"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[main] ERROR: Input CSV is missing columns: {missing}")
        sys.exit(1)

    # Initialise agent
    try:
        agent = TriageAgent(retriever=retriever)
    except EnvironmentError as exc:
        print(f"[main] ERROR: {exc}")
        sys.exit(1)

    # Write log header
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(
            f"\n{'#'*60}\n"
            f"# RUN STARTED: {datetime.now().isoformat()}\n"
            f"# INPUT: {input_path}\n"
            f"{'#'*60}\n"
        )

    # Process tickets
    results = []
    print(f"[main] Processing {len(df)} tickets from: {input_path}\n")

    for i, row in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc="Triaging tickets", unit="ticket"),
        start=1,
    ):
        issue = str(row.Issue) if pd.notna(row.Issue) else ""
        subject = str(row.Subject) if pd.notna(row.Subject) else ""
        company = str(row.Company) if pd.notna(row.Company) else "None"

        try:
            result = agent.process(issue=issue, subject=subject, company=company)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).error(
                "Ticket #%d failed unexpectedly: %s", i, exc
            )
            result = {
                "status": "escalated",
                "product_area": "Unknown",
                "response": "An internal error occurred. Please contact support.",
                "justification": f"Processing error: {exc}",
                "request_type": "invalid",
                "retrieved_docs": [],
            }

        # Log ticket
        _write_ticket_log(
            ticket_num=i,
            issue=issue,
            company=company,
            decision=result["status"],
            retrieved_docs=result["retrieved_docs"],
            response=result["response"],
            justification=result["justification"],
        )

        results.append(
            {
                "Issue": issue,
                "Subject": subject,
                "Company": company,
                "status": result["status"],
                "product_area": result["product_area"],
                "response": result["response"],
                "justification": result["justification"],
                "request_type": result["request_type"],
            }
        )

    # Write output CSV
    out_df = pd.DataFrame(results, columns=[
        "Issue", "Subject", "Company",
        "status", "product_area", "response", "justification", "request_type",
    ])
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n[main] Done! Results written to: {OUTPUT_CSV}")
    print(f"[main] Full log: {LOG_FILE}\n")

    # Summary stats
    replied = (out_df["status"] == "replied").sum()
    escalated = (out_df["status"] == "escalated").sum()
    print(f"  Tickets processed : {len(out_df)}")
    print(f"  Replied           : {replied}")
    print(f"  Escalated         : {escalated}")


# ---------------------------------------------------------------------------
# Retriever import (forward reference helper)
# ---------------------------------------------------------------------------

# Imported inside functions to avoid circular imports at module load time.
Retriever = None  # placeholder to satisfy type checker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Domain Support Triage Agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Crawl support sites and save to corpus/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-crawl even if corpus already contains files",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        dest="build_index",
        help="Build the TF-IDF index from corpus/",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Process tickets and write output.csv",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="FILE",
        help=f"Input CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run crawl + build-index + run in sequence",
    )
    return parser


def main() -> None:
    _setup_logging()
    logger = logging.getLogger("main")

    parser = build_parser()
    args = parser.parse_args()

    # Show help if no arguments given
    if not any([args.crawl, args.build_index, args.run, args.all]):
        parser.print_help()
        sys.exit(0)

    print(f"\n{'='*60}")
    print("  MULTI-DOMAIN SUPPORT TRIAGE AGENT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    retriever_instance = None

    # --all = crawl + build-index + run
    if args.all:
        cmd_crawl(force=args.force)
        retriever_instance = cmd_build_index()
        cmd_run(input_path=args.input, retriever=retriever_instance)
        return

    if args.crawl:
        cmd_crawl(force=args.force)

    if args.build_index:
        retriever_instance = cmd_build_index()

    if args.run:
        if retriever_instance is None:
            # Build the index automatically if not already done
            retriever_instance = cmd_build_index()
        cmd_run(input_path=args.input, retriever=retriever_instance)


if __name__ == "__main__":
    main()
