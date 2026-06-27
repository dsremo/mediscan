import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mediscan_core import TriageResult, UrgencyLevel
from safety import detect_red_flags, escalate_for_safety


def _result(urgency):
    return TriageResult("rash", 0.6, urgency, "desc", ["wash"], "see a worker", ["since when?"])


def test_detect():
    assert detect_red_flags("the rash is spreading rapidly with red streaks and high fever")
    assert detect_red_flags("patient has difficulty breathing")
    assert not detect_red_flags("small dry patch on the elbow, mild itch")


def test_escalates_low_to_emergency():
    result, flags = escalate_for_safety(_result(UrgencyLevel.LOW), "spreading rapidly, high fever, confusion")
    assert result.urgency == UrgencyLevel.EMERGENCY
    assert flags
    assert "EMERGENCY RED FLAG" in result.when_to_seek_care


def test_no_change_when_clean():
    result, flags = escalate_for_safety(_result(UrgencyLevel.LOW), "mild dry skin")
    assert result.urgency == UrgencyLevel.LOW
    assert flags == []
