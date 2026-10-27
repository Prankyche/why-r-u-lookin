import pytest

from main import parse_target


def test_target_none_means_never_found():
    assert parse_target("none") is None and parse_target(" NONE ") is None


def test_target_pair():
    assert parse_target("25,21") == (25.0, 21.0) and parse_target(" 3.5 , 8 ") == (3.5, 8.0)


@pytest.mark.parametrize("bad", ["25", "a,b", "1,2,3", ""])
def test_bad_target_exits_with_a_message(bad):
    with pytest.raises(SystemExit) as e:
        parse_target(bad)
    assert "--sitl-target" in str(e.value)