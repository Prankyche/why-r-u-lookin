from fsm.mission_fsm import Event, MissionFSM, State


def to_descend(zone_first):
    f = MissionFSM()
    for e in [Event.ARMED, Event.ALT_REACHED, Event.QR_VALID, Event.BANNER_FOUND,
              Event.CORRIDOR_EXIT, Event.ALT_REACHED]:
        f.step(e)
    assert f.state is State.SEARCH_QR
    if zone_first:
        f.step(Event.IN_ZONE)
    f.step(Event.QR_MATCH)
    assert f.state is State.DESCEND_DROP
    return f


def test_gate_opens_when_altitude_arrives_last():
    f = to_descend(zone_first=True)
    f.step(Event.ALT_REACHED)
    assert f.state is State.RELEASE


def test_gate_opens_when_zone_arrives_after_altitude():       # the old bug: this used to hold until timeout
    f = to_descend(zone_first=False)
    f.step(Event.ALT_REACHED)
    assert f.state is State.DESCEND_DROP
    f.step(Event.IN_ZONE)
    assert f.state is State.RELEASE


def test_gate_stays_shut_without_zone():
    f = to_descend(zone_first=False)
    for _ in range(5):
        f.step(Event.ALT_REACHED)
    assert f.state is State.DESCEND_DROP


def test_altitude_latch_resets_on_reentry():
    f = to_descend(zone_first=False)
    f.step(Event.ALT_REACHED)
    assert f.alt_ok
    f._goto(State.DESCEND_DROP, reason="test")
    assert not f.alt_ok


def test_held_gate_logs_once_per_change_not_every_tick(caplog):
    f = to_descend(zone_first=False)
    with caplog.at_level("INFO", logger="mission_fsm"):
        for _ in range(30):
            f.step(Event.ALT_REACHED)
    assert sum("Drop gate held" in r.message for r in caplog.records) == 1