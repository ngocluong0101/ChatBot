"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

from evals.perf.summarize import perf_table


def report(label, users=1, rps=0.5, errors=None):
    return {
        "meta": {"label": label},
        "levels": [
            {
                "users": users,
                "requests": 20,
                "throughput_rps": rps,
                "latency_seconds": {"p50": 2.04, "p95": 3.85},
                "ttft_seconds": {"p50": 1.0},
                "errors": errors or {},
                "gpu": {"vram_peak_mib": 6788},
            }
        ],
    }


def test_table_orders_baseline_first_and_copies_numbers_from_the_reports():
    table = perf_table([report("parallel-4", 20, 1.25), report("baseline", 1, 0.25)])
    lines = table.splitlines()
    assert lines[2].startswith("| baseline | 1 | 20 | 0.25 |")
    assert lines[3].startswith("| parallel-4 | 20 | 20 | 1.25 | 2.0 | 3.9 | 1.0 | 0 | 6788 |")


def test_table_shows_error_breakdown_and_missing_values():
    broken = report("serial-1", errors={"http_503": 3, "timeout": 1})
    broken["levels"][0]["latency_seconds"] = {"p50": None, "p95": None}
    broken["levels"][0]["gpu"] = None
    row = perf_table([broken]).splitlines()[2]
    assert "3 http_503, 1 timeout" in row and "| – | – |" in row and row.endswith("| – |")
