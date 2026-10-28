"""Interactive terminal progress display."""

from io import StringIO

from deh.cli.progress import ProgressBars


def test_counted_notices_are_replaced_but_warnings_remain():
    stream = StringIO()
    bars = ProgressBars(stream=stream)
    bars.update("features", 0, 5)
    bars.update("features", 5, 5)
    bars.log("[Features] Extracted features for 5/5 images")
    bars.log("[Hash] Using max Hamming distance: 16 (equivalent to 75% similarity)")
    bars.log("[Hash] Found 1 groups containing 2 similar images")
    bars.log("      [timing] Perceptual hash verification: 0.01s")
    bars.log("[Features] Skipped 1 image(s): unreadable")
    bars.log("[Phase 3b] Verifying deep matches with perceptual hashes...")
    bars.close()

    output = stream.getvalue()
    assert "Extracting features" in output
    assert "Extracted features for 5/5" not in output
    assert "Using max Hamming distance" not in output
    assert "Found 1 groups" not in output
    assert "Perceptual hash verification: 0.01s" not in output
    assert "Skipped 1 image(s)" in output
    assert "Verifying deep matches" in output
