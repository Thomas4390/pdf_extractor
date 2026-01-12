#!/usr/bin/env python3
"""
Test script for comparing VLM model extraction quality and extraction modes.

Usage:
    python -m src.tests.test_model_comparison                    # Run ALL sources (default)
    python -m src.tests.test_model_comparison <source>           # Run single source
    python -m src.tests.test_model_comparison <source> --mode <mode>
    python -m src.tests.test_model_comparison <source> --compare-modes
    python -m src.tests.test_model_comparison <source> --model <model_id>

Sources: UV, IDC, IDC_STATEMENT, ASSOMPTION

Options:
    --single         Run comparison on a single source only (default: UV)
    --invalidate     Clear cache before running to force fresh extraction
    --mode <mode>    Extraction mode: vision, text, pdf_native, hybrid
    --compare-modes  Compare all extraction modes for a single model
    --model <id>     Specify model(s) to test (can be used multiple times)
                     Use short name (qwen3-vl) or full ID (qwen/qwen3-vl-235b-a22b-instruct)

Examples:
    python -m src.tests.test_model_comparison              # Test ALL sources (default)
    python -m src.tests.test_model_comparison UV           # Test UV only
    python -m src.tests.test_model_comparison UV --mode hybrid  # Test UV with hybrid mode
    python -m src.tests.test_model_comparison UV --compare-modes  # Compare all modes on UV
    python -m src.tests.test_model_comparison UV --model deepseek/deepseek-chat
    python -m src.tests.test_model_comparison UV --model qwen3-vl --model gemini-3-flash
    python -m src.tests.test_model_comparison UV --mode hybrid --model deepseek/deepseek-chat
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Models to compare (default if no --model specified)
MODELS_TO_TEST = {
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "qwen3-vl": "qwen/qwen3-vl-235b-a22b-instruct",
    # "qwen2.5-vl": "qwen/qwen2.5-vl-72b-instruct",
}

# Model aliases for short names
MODEL_ALIASES = {
    "qwen3-vl": "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen2.5-vl": "qwen/qwen2.5-vl-72b-instruct",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "gemini-2-flash": "google/gemini-2.0-flash-001",
    "deepseek": "deepseek/deepseek-chat",
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-r1",
    "claude-sonnet": "anthropic/claude-sonnet-4",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}

# Extraction modes to compare (for --compare-modes)
MODES_TO_TEST = ["vision", "pdf_native", "hybrid"]

# Default model for mode comparison
DEFAULT_MODE_COMPARISON_MODEL = "deepseek/deepseek-chat"

# Test PDFs by source
TEST_PDFS = {
    "UV": PROJECT_ROOT / "pdf/uv/rappportremun_21621_2025-12-08.pdf",
    "IDC": PROJECT_ROOT / "pdf/idc/Rapport des propositions soumises.20251217_0707.pdf",
    "ASSOMPTION": PROJECT_ROOT / "pdf/assomption/Remuneration - 2025-12-05T133703.562.pdf",
    "IDC_STATEMENT": PROJECT_ROOT / "pdf/idc_statement/Statements (17).pdf",
}


# =============================================================================
# TERMINAL COLORS AND STYLING
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


c = Colors()

# Box drawing characters
BOX = {
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
    "h": "─", "v": "│",
    "lj": "├", "rj": "┤", "tj": "┬", "bj": "┴", "x": "┼",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractionResult:
    """Result of a single model extraction."""
    model_name: str
    model_id: str
    result: Optional[dict] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    item_count: int = 0
    cost: float = 0.0
    tokens: int = 0
    mode: str = "vision"

    @property
    def success(self) -> bool:
        return self.result is not None


@dataclass
class ComparisonReport:
    """Full comparison report for a source."""
    source: str
    pdf_path: Path
    results: list[ExtractionResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.results)

    @property
    def total_tokens(self) -> float:
        return sum(r.tokens for r in self.results)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_box(title: str, content: list[str], width: int = 70, color: str = c.CYAN):
    """Print content in a styled box."""
    inner_width = width - 2

    # Top border
    print(f"{color}{BOX['tl']}{BOX['h'] * inner_width}{BOX['tr']}{c.RESET}")

    # Title
    if title:
        title_display = f" {title} "
        padding = inner_width - len(title_display)
        left_pad = padding // 2
        right_pad = padding - left_pad
        print(f"{color}{BOX['v']}{c.RESET}{' ' * left_pad}{c.BOLD}{title_display}{c.RESET}{' ' * right_pad}{color}{BOX['v']}{c.RESET}")
        print(f"{color}{BOX['lj']}{BOX['h'] * inner_width}{BOX['rj']}{c.RESET}")

    # Content
    for line in content:
        # Handle ANSI codes when calculating padding
        visible_len = len(line.encode().decode('unicode_escape').replace('\033[', '\x1b['))
        # Crude estimate - count visible characters
        import re
        visible_line = re.sub(r'\033\[[0-9;]*m', '', line)
        padding = inner_width - len(visible_line)
        print(f"{color}{BOX['v']}{c.RESET} {line}{' ' * max(0, padding - 1)}{color}{BOX['v']}{c.RESET}")

    # Bottom border
    print(f"{color}{BOX['bl']}{BOX['h'] * inner_width}{BOX['br']}{c.RESET}")


def print_header(title: str, subtitle: str = ""):
    """Print a large header."""
    width = 70
    print()
    print(f"{c.BRIGHT_CYAN}{'═' * width}{c.RESET}")
    print(f"{c.BRIGHT_CYAN}║{c.RESET} {c.BOLD}{title}{c.RESET}")
    if subtitle:
        print(f"{c.BRIGHT_CYAN}║{c.RESET} {c.DIM}{subtitle}{c.RESET}")
    print(f"{c.BRIGHT_CYAN}{'═' * width}{c.RESET}")
    print()


def print_section(title: str):
    """Print a section header."""
    print(f"\n{c.YELLOW}▸ {title}{c.RESET}")
    print(f"{c.GRAY}{'─' * 50}{c.RESET}")


def progress_bar(current: int, total: int, width: int = 30, label: str = "") -> str:
    """Generate a progress bar string."""
    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = (current / total * 100) if total > 0 else 0
    return f"{c.CYAN}[{bar}]{c.RESET} {pct:5.1f}% {label}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def format_cost(cost: float) -> str:
    """Format cost in dollars."""
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def resolve_model(model_input: str) -> tuple[str, str]:
    """
    Resolve a model input to (display_name, model_id).

    Args:
        model_input: Short name (qwen3-vl) or full ID (qwen/qwen3-vl-235b-a22b-instruct)

    Returns:
        Tuple of (display_name, full_model_id)
    """
    # Check if it's an alias
    if model_input in MODEL_ALIASES:
        return model_input, MODEL_ALIASES[model_input]

    # Check if it's already a full model ID (contains /)
    if "/" in model_input:
        # Extract short name from full ID
        short_name = model_input.split("/")[-1]
        # Simplify long names
        if len(short_name) > 20:
            short_name = short_name[:17] + "..."
        return short_name, model_input

    # Unknown model, use as-is
    return model_input, model_input


def parse_model_args(args: list[str]) -> tuple[dict[str, str], list[str]]:
    """
    Parse --model arguments from command line.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (models_dict, remaining_args)
    """
    models = {}
    remaining = []
    i = 0

    while i < len(args):
        if args[i] == "--model":
            if i + 1 < len(args):
                model_input = args[i + 1]
                name, model_id = resolve_model(model_input)
                models[name] = model_id
                i += 2
            else:
                print(f"{c.BRIGHT_RED}Error:{c.RESET} --model requires a value")
                sys.exit(1)
        else:
            remaining.append(args[i])
            i += 1

    return models, remaining


def status_icon(success: bool) -> str:
    """Get status icon."""
    return f"{c.BRIGHT_GREEN}✓{c.RESET}" if success else f"{c.BRIGHT_RED}✗{c.RESET}"


# =============================================================================
# EXTRACTION LOGIC
# =============================================================================

async def extract_with_model(
    pdf_path: Path,
    source: str,
    model_name: str,
    model_id: str,
    mode: str = "vision",
) -> ExtractionResult:
    """
    Extract data from PDF using a specific model and extraction mode.

    Args:
        pdf_path: Path to the PDF file
        source: Document type (UV, IDC, etc.)
        model_name: Display name for the model
        model_id: OpenRouter model identifier
        mode: Extraction mode (vision, text, pdf_native, hybrid)

    Returns:
        ExtractionResult with all metrics
    """
    from src.utils.model_registry import (
        register_model,
        get_model_config,
        ModelConfig,
        ExtractionMode,
        OcrEngine,
    )
    from src.extractors import (
        UVExtractor,
        IDCExtractor,
        AssomptionExtractor,
        IDCStatementExtractor,
    )

    result = ExtractionResult(model_name=model_name, model_id=model_id, mode=mode)

    # Get original config
    original_config = get_model_config(source)

    # Map mode string to ExtractionMode enum
    mode_map = {
        "vision": ExtractionMode.VISION,
        "text": ExtractionMode.TEXT,
        "pdf_native": ExtractionMode.PDF_NATIVE,
        "hybrid": ExtractionMode.HYBRID,
    }
    extraction_mode = mode_map.get(mode, ExtractionMode.VISION)

    # Register test model with the specified mode
    test_config = ModelConfig(
        model_id=model_id,
        mode=extraction_mode,
        fallback_model_id=None,  # No fallback for comparison
        fallback_mode=None,
        page_config=original_config.page_config,
        ocr_engine=OcrEngine.MISTRAL_OCR,
        text_analysis_model="deepseek/deepseek-chat",
    )
    register_model(source, test_config)

    # Select extractor
    extractors = {
        "UV": UVExtractor,
        "IDC": IDCExtractor,
        "ASSOMPTION": AssomptionExtractor,
        "IDC_STATEMENT": IDCStatementExtractor,
    }

    extractor_class = extractors.get(source)
    if not extractor_class:
        result.error = f"Unknown source: {source}"
        return result

    extractor = extractor_class()

    mode_label = f"{c.MAGENTA}{mode}{c.RESET}" if mode != "vision" else mode
    print(f"   {c.DIM}[{model_name}]{c.RESET} {c.DIM}({mode_label}){c.RESET} Extracting... ", end="", flush=True)

    start_time = time.time()
    try:
        # Use extract_raw to get dict for comparison
        extraction = await extractor.extract_raw(pdf_path)
        result.elapsed_seconds = time.time() - start_time
        result.result = extraction

        # Count items based on source
        if source == "UV":
            result.item_count = len(extraction.get("activites", []))
        elif source == "IDC":
            result.item_count = len(extraction.get("propositions", []))
        elif source == "ASSOMPTION":
            result.item_count = len(extraction.get("commissions", []))
        elif source == "IDC_STATEMENT":
            result.item_count = len(extraction.get("trailing_fees", []))

        # Get cost from session (if tracked)
        session = extractor.client.get_session_summary()
        result.cost = session.get("total_cost", 0)
        result.tokens = session.get("total_prompt_tokens", 0) + session.get("total_completion_tokens", 0)

        print(f"{c.BRIGHT_GREEN}✓{c.RESET} {format_duration(result.elapsed_seconds)} | {result.item_count} items")

    except Exception as e:
        result.elapsed_seconds = time.time() - start_time
        result.error = str(e)
        print(f"{c.BRIGHT_RED}✗{c.RESET} {format_duration(result.elapsed_seconds)} | Error: {e}")

    finally:
        # Restore original config
        register_model(source, original_config)

    return result


# =============================================================================
# COMPARISON DISPLAY
# =============================================================================

def display_comparison_table(report: ComparisonReport, show_mode: bool = False):
    """Display a comparison table for results."""
    print_section(f"Results Summary: {report.source}")

    # Check if we have different modes in the results
    has_modes = show_mode or len(set(r.mode for r in report.results)) > 1

    # Header
    if has_modes:
        print(f"\n   {c.BOLD}{'Model':<18} {'Mode':<12} {'Status':<10} {'Time':<10} {'Items':<8} {'Cost':<12}{c.RESET}")
        print(f"   {c.GRAY}{'─' * 70}{c.RESET}")
    else:
        print(f"\n   {c.BOLD}{'Model':<18} {'Status':<10} {'Time':<10} {'Items':<8} {'Cost':<12}{c.RESET}")
        print(f"   {c.GRAY}{'─' * 58}{c.RESET}")

    for r in report.results:
        status = f"{c.BRIGHT_GREEN}Success{c.RESET}" if r.success else f"{c.BRIGHT_RED}Failed{c.RESET}"
        time_str = format_duration(r.elapsed_seconds)
        items_str = str(r.item_count) if r.success else "-"
        cost_str = format_cost(r.cost) if r.cost > 0 else f"{c.DIM}N/A{c.RESET}"

        if has_modes:
            mode_str = f"{c.MAGENTA}{r.mode}{c.RESET}" if r.mode != "vision" else r.mode
            print(f"   {r.model_name:<18} {mode_str:<21} {status:<19} {time_str:<10} {items_str:<8} {cost_str:<12}")
        else:
            print(f"   {r.model_name:<18} {status:<19} {time_str:<10} {items_str:<8} {cost_str:<12}")

    if has_modes:
        print(f"   {c.GRAY}{'─' * 70}{c.RESET}")
    else:
        print(f"   {c.GRAY}{'─' * 58}{c.RESET}")

    # Total
    total_time = sum(r.elapsed_seconds for r in report.results)
    if has_modes:
        print(f"   {c.BOLD}{'TOTAL':<18}{c.RESET} {'':<12} {'':<10} {format_duration(total_time):<10} {'':<8} {format_cost(report.total_cost):<12}")
    else:
        print(f"   {c.BOLD}{'TOTAL':<18}{c.RESET} {'':<10} {format_duration(total_time):<10} {'':<8} {format_cost(report.total_cost):<12}")


def display_detailed_comparison(report: ComparisonReport):
    """Display detailed field-by-field comparison showing ALL differences."""
    successful = [r for r in report.results if r.success]

    if len(successful) < 2:
        return

    print_section("Detailed Comparison")

    # Check item count consistency
    counts = {r.model_name: r.item_count for r in successful}
    unique_counts = set(counts.values())

    if len(unique_counts) == 1:
        print(f"   {c.BRIGHT_GREEN}✓{c.RESET} Item counts match: {c.BOLD}{list(unique_counts)[0]}{c.RESET} items")
    else:
        print(f"   {c.BRIGHT_YELLOW}⚠{c.RESET} Item counts differ:")
        for model, count in counts.items():
            print(f"      • {model}: {count} items")

    # Get all items from each result
    all_items = {}
    for r in successful:
        if r.result:
            if report.source == "UV":
                items = r.result.get("activites", [])
            elif report.source == "IDC":
                items = r.result.get("propositions", [])
            elif report.source == "ASSOMPTION":
                items = r.result.get("commissions", [])
            elif report.source == "IDC_STATEMENT":
                items = r.result.get("trailing_fees", [])
            else:
                items = []
            all_items[r.model_name] = items

    if len(all_items) < 2:
        return

    # Find minimum item count across all models
    min_items = min(len(items) for items in all_items.values())
    if min_items == 0:
        print(f"\n   {c.BRIGHT_YELLOW}⚠{c.RESET} No items to compare")
        return

    # Track all differences
    differences = []
    matches = 0
    total_fields = 0

    # Compare each item
    for item_idx in range(min_items):
        items_at_idx = {model: items[item_idx] for model, items in all_items.items() if item_idx < len(items)}

        # Get all keys from items at this index
        all_keys = set()
        for item in items_at_idx.values():
            if isinstance(item, dict):
                all_keys.update(item.keys())

        # Compare each field
        for key in sorted(all_keys):
            total_fields += 1
            values = {}
            for model, item in items_at_idx.items():
                if isinstance(item, dict):
                    values[model] = item.get(key, "N/A")
                else:
                    values[model] = "N/A"

            unique_values = set(str(v) for v in values.values())

            if len(unique_values) == 1:
                matches += 1
            else:
                differences.append({
                    "item_idx": item_idx,
                    "field": key,
                    "values": values,
                })

    # Display summary
    match_rate = (matches / total_fields * 100) if total_fields > 0 else 0
    print(f"\n   {c.BOLD}Field Comparison Summary:{c.RESET}")
    print(f"   Items compared: {min_items}")
    print(f"   Total fields: {total_fields}")
    print(f"   Matches: {c.GREEN}{matches}{c.RESET} ({match_rate:.1f}%)")
    print(f"   Differences: {c.YELLOW}{len(differences)}{c.RESET}")

    # Display all differences
    if differences:
        print(f"\n   {c.BOLD}{c.YELLOW}All Differences ({len(differences)}):{c.RESET}")
        print(f"   {c.GRAY}{'─' * 60}{c.RESET}")

        for diff in differences:
            item_idx = diff["item_idx"]
            field = diff["field"]
            values = diff["values"]

            print(f"\n   {c.YELLOW}≠{c.RESET} {c.CYAN}[Item {item_idx + 1}] {field}{c.RESET}:")
            for model, value in values.items():
                val_str = str(value)
                # Truncate long values but show more than before
                if len(val_str) > 80:
                    val_str = val_str[:80] + "..."
                print(f"      {c.DIM}{model}:{c.RESET} {val_str}")

        print(f"\n   {c.GRAY}{'─' * 60}{c.RESET}")
    else:
        print(f"\n   {c.BRIGHT_GREEN}✓ All fields match across all models!{c.RESET}")


def save_results(report: ComparisonReport):
    """Save results to JSON files."""
    output_dir = PROJECT_ROOT / "cache" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("Saved Results")

    for r in report.results:
        if r.result:
            output_file = output_dir / f"{report.source}_{r.model_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(r.result, f, indent=2, ensure_ascii=False, default=str)
            print(f"   {c.DIM}→{c.RESET} {output_file.name}")


# =============================================================================
# MAIN COMPARISON RUNNERS
# =============================================================================

async def run_single_comparison(
    source: str,
    invalidate_cache: bool = False,
    mode: str = "vision",
    models: Optional[dict[str, str]] = None,
) -> Optional[ComparisonReport]:
    """Run model comparison for a single source.

    Args:
        source: Document source type (UV, IDC, etc.)
        invalidate_cache: Clear cache before extraction
        mode: Extraction mode (vision, text, pdf_native, hybrid)
        models: Dict of {display_name: model_id} to test. Defaults to MODELS_TO_TEST.
    """
    # Use provided models or default
    models_to_use = models if models else MODELS_TO_TEST

    pdf_path = TEST_PDFS.get(source)

    if not pdf_path:
        print(f"{c.BRIGHT_RED}Error:{c.RESET} Unknown source: {source}")
        print(f"Available sources: {', '.join(TEST_PDFS.keys())}")
        return None

    if not pdf_path.exists():
        print(f"{c.BRIGHT_RED}Error:{c.RESET} PDF not found: {pdf_path}")
        return None

    mode_display = f" ({c.MAGENTA}{mode}{c.RESET})" if mode != "vision" else ""
    print_header(
        f"Model Comparison: {source}{mode_display}",
        f"PDF: {pdf_path.name}"
    )

    # Invalidate cache if requested
    if invalidate_cache:
        from src.extractors import (
            UVExtractor,
            IDCExtractor,
            AssomptionExtractor,
            IDCStatementExtractor,
        )

        extractors = {
            "UV": UVExtractor,
            "IDC": IDCExtractor,
            "ASSOMPTION": AssomptionExtractor,
            "IDC_STATEMENT": IDCStatementExtractor,
        }

        extractor = extractors[source]()
        if extractor.is_cached(pdf_path):
            extractor.invalidate_cache(pdf_path)
            print(f"   {c.YELLOW}⟳{c.RESET} Cache invalidated")

    print_section("Running Extractions")
    print(f"   Testing {len(models_to_use)} model(s): {', '.join(models_to_use.keys())}")
    print(f"   Mode: {c.MAGENTA}{mode}{c.RESET}\n")

    report = ComparisonReport(source=source, pdf_path=pdf_path)

    for model_name, model_id in models_to_use.items():
        result = await extract_with_model(pdf_path, source, model_name, model_id, mode=mode)
        report.results.append(result)

    # Display results
    display_comparison_table(report, show_mode=(mode != "vision"))
    display_detailed_comparison(report)
    save_results(report)

    return report


async def run_mode_comparison(source: str, invalidate_cache: bool = False) -> Optional[ComparisonReport]:
    """
    Compare different extraction modes for a single source using a default model.

    This tests vision, pdf_native, and hybrid modes to compare:
    - Extraction quality (item count, field accuracy)
    - Cost efficiency
    - Processing time
    """
    pdf_path = TEST_PDFS.get(source)

    if not pdf_path:
        print(f"{c.BRIGHT_RED}Error:{c.RESET} Unknown source: {source}")
        print(f"Available sources: {', '.join(TEST_PDFS.keys())}")
        return None

    if not pdf_path.exists():
        print(f"{c.BRIGHT_RED}Error:{c.RESET} PDF not found: {pdf_path}")
        return None

    print_header(
        f"Mode Comparison: {source}",
        f"PDF: {pdf_path.name}"
    )

    # Invalidate cache if requested
    if invalidate_cache:
        from src.extractors import (
            UVExtractor,
            IDCExtractor,
            AssomptionExtractor,
            IDCStatementExtractor,
        )

        extractors = {
            "UV": UVExtractor,
            "IDC": IDCExtractor,
            "ASSOMPTION": AssomptionExtractor,
            "IDC_STATEMENT": IDCStatementExtractor,
        }

        extractor = extractors[source]()
        if extractor.is_cached(pdf_path):
            extractor.invalidate_cache(pdf_path)
            print(f"   {c.YELLOW}⟳{c.RESET} Cache invalidated")

    print_section("Running Mode Comparisons")
    print(f"   Testing {len(MODES_TO_TEST)} modes: {', '.join(MODES_TO_TEST)}")
    print(f"   Model: {c.CYAN}{DEFAULT_MODE_COMPARISON_MODEL}{c.RESET}\n")

    report = ComparisonReport(source=source, pdf_path=pdf_path)

    # For mode comparison, use a single model but different modes
    model_name = "deepseek"
    model_id = DEFAULT_MODE_COMPARISON_MODEL

    for mode in MODES_TO_TEST:
        result = await extract_with_model(
            pdf_path, source, f"{model_name}-{mode}", model_id, mode=mode
        )
        report.results.append(result)

    # Display results with mode column
    display_comparison_table(report, show_mode=True)
    display_detailed_comparison(report)

    # Mode-specific summary
    print_section("Mode Comparison Summary")
    for r in report.results:
        if r.success:
            cost_per_item = r.cost / r.item_count if r.item_count > 0 else 0
            print(f"   {c.MAGENTA}{r.mode:<12}{c.RESET} {r.item_count:>4} items | {format_cost(r.cost):>12} | {format_cost(cost_per_item)}/item | {format_duration(r.elapsed_seconds)}")
        else:
            print(f"   {c.MAGENTA}{r.mode:<12}{c.RESET} {c.BRIGHT_RED}FAILED{c.RESET}")

    save_results(report)

    return report


async def run_all_comparisons(
    invalidate_cache: bool = False,
    mode: str = "vision",
    models: Optional[dict[str, str]] = None,
) -> list[ComparisonReport]:
    """Run model comparison for all sources.

    Args:
        invalidate_cache: Clear cache before extraction
        mode: Extraction mode (vision, text, pdf_native, hybrid)
        models: Dict of {display_name: model_id} to test. Defaults to MODELS_TO_TEST.
    """
    models_to_use = models if models else MODELS_TO_TEST

    mode_display = f" ({c.MAGENTA}{mode}{c.RESET})" if mode != "vision" else ""
    print_header(
        f"Full Model Comparison{mode_display}",
        f"Testing {len(TEST_PDFS)} sources with {len(models_to_use)} model(s)"
    )

    reports = []
    total_cost = 0.0

    for i, source in enumerate(TEST_PDFS.keys(), 1):
        print(f"\n{c.BRIGHT_MAGENTA}{'━' * 70}{c.RESET}")
        print(f"{c.BRIGHT_MAGENTA}  [{i}/{len(TEST_PDFS)}] Processing: {source}{c.RESET}")
        print(f"{c.BRIGHT_MAGENTA}{'━' * 70}{c.RESET}")

        report = await run_single_comparison(source, invalidate_cache, mode, models_to_use)
        if report:
            reports.append(report)
            total_cost += report.total_cost

    # Final summary
    print_header("Final Summary", f"Completed {len(reports)} source comparisons")

    print_section("Cost Summary by Source")
    for report in reports:
        print(f"   {report.source:<15} {format_cost(report.total_cost)}")

    print(f"\n   {c.BOLD}{'GRAND TOTAL':<15} {format_cost(total_cost)}{c.RESET}")

    # Success rate
    print_section("Success Rate by Model")
    for model_name in models_to_use.keys():
        successes = sum(1 for r in reports for res in r.results if res.model_name == model_name and res.success)
        total = len(reports)
        bar = progress_bar(successes, total, width=20)
        print(f"   {model_name:<18} {bar} ({successes}/{total})")

    return reports


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    """Main entry point."""
    args = sys.argv[1:]

    # Parse --model arguments first (extracts them from args)
    custom_models, args = parse_model_args(args)

    # Parse flags
    invalidate_cache = "--invalidate" in args
    single_source = "--single" in args  # Use --single to test only one source
    compare_modes = "--compare-modes" in args

    # Parse --mode <value>
    mode = "vision"
    if "--mode" in args:
        mode_idx = args.index("--mode")
        if mode_idx + 1 < len(args):
            mode = args[mode_idx + 1].lower()
            args = args[:mode_idx] + args[mode_idx + 2:]
        else:
            print(f"{c.BRIGHT_RED}Error:{c.RESET} --mode requires a value (vision, text, pdf_native, hybrid)")
            sys.exit(1)

    # Filter out flags
    args = [a for a in args if not a.startswith("--")]

    # Validate mode
    valid_modes = ["vision", "text", "pdf_native", "hybrid"]
    if mode not in valid_modes:
        print(f"{c.BRIGHT_RED}Error:{c.RESET} Invalid mode: {mode}")
        print(f"Valid modes: {', '.join(valid_modes)}")
        sys.exit(1)

    # Use custom models if provided, otherwise use defaults
    models = custom_models if custom_models else None

    if compare_modes:
        source = args[0].upper() if args else "UV"

        valid_sources = list(TEST_PDFS.keys())
        if source not in valid_sources:
            print(f"{c.BRIGHT_RED}Error:{c.RESET} Invalid source: {source}")
            print(f"Valid sources: {', '.join(valid_sources)}")
            sys.exit(1)

        await run_mode_comparison(source, invalidate_cache)
    elif single_source or args:
        # If --single flag or a specific source is provided, run single comparison
        source = args[0].upper() if args else "UV"

        valid_sources = list(TEST_PDFS.keys())
        if source not in valid_sources:
            print(f"{c.BRIGHT_RED}Error:{c.RESET} Invalid source: {source}")
            print(f"Valid sources: {', '.join(valid_sources)}")
            print(f"\nUsage: python -m src.tests.test_model_comparison [source] [--single] [--invalidate] [--mode <mode>] [--compare-modes] [--model <id>]")
            sys.exit(1)

        await run_single_comparison(source, invalidate_cache, mode, models)
    else:
        # Default: run all comparisons
        await run_all_comparisons(invalidate_cache, mode, models)


if __name__ == "__main__":
    asyncio.run(main())
