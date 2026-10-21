import pytest
from fsm.mission_fsm import MissionFSM, State, Event


def run(fsm, events):
    for e in events:
        fsm.step(e)
    return fsm.state


def reach_search(fsm):
    """Drive the FSM from IDLE to SEARCH_QR through the normal flow."""
    run(fsm, [
        Event.ARMED,
        Event.ALT_REACHED,        # takeoff -> scan
        Event.QR_VALID,           # scan -> align
        Event.BANNER_FOUND,       # align -> corridor
        Event.CORRIDOR_EXIT,      # corridor -> ascend
        Event.ALT_REACHED,        # ascend -> search
    ])
    assert fsm.state is State.SEARCH_QR


def test_happy_path_reaches_done():
    fsm = MissionFSM()
    reach_search(fsm)
    run(fsm, [
        Event.QR_MATCH,           # search -> descend
        Event.IN_ZONE,            # latch zone
        Event.ALT_REACHED,        # descend -> release
        Event.DROP_CONFIRMED,     # release -> ascend return
        Event.ALT_REACHED,        # ascend return -> return corridor
        Event.CORRIDOR_EXIT,      # return corridor -> land
        Event.LANDED,             # land -> done
    ])
    assert fsm.state is State.DONE
    assert fsm.is_done()


def test_search_loops_on_mismatch_then_matches():
    fsm = MissionFSM()
    reach_search(fsm)
    run(fsm, [Event.QR_MISMATCH, Event.QR_MISMATCH, Event.QR_MISMATCH])
    assert fsm.state is State.SEARCH_QR
    run(fsm, [Event.QR_MATCH])
    assert fsm.state is State.DESCEND_DROP


def test_drop_gate_holds_without_zone():
    fsm = MissionFSM()
    reach_search(fsm)
    run(fsm, [Event.QR_MATCH, Event.ALT_REACHED])   # no IN_ZONE
    assert fsm.state is State.DESCEND_DROP          # held, did not release


def test_drop_gate_holds_without_qr_match():
    fsm = MissionFSM()
    fsm.state = State.DESCEND_DROP
    fsm.zone_ok = True
    fsm.step(Event.ALT_REACHED)
    assert fsm.state is State.DESCEND_DROP


def test_link_lost_mid_corridor_goes_to_rtl():
    fsm = MissionFSM()
    run(fsm, [Event.ARMED, Event.ALT_REACHED, Event.QR_VALID,
              Event.BANNER_FOUND])
    fsm.step(Event.LINK_LOST)
    assert fsm.state is State.RTL
    assert fsm.is_done()


@pytest.mark.parametrize("event", [Event.LOW_BATT, Event.GEOFENCE_BREACH,
                                   Event.UNSAFE, Event.PILOT_TAKEOVER])
def test_failsafe_from_any_state(event):
    fsm = MissionFSM()
    fsm.step(Event.ARMED)
    fsm.step(event)
    assert fsm.state is State.RTL


def test_timeout_triggers_rtl(monkeypatch):
    fsm = MissionFSM()
    fsm.step(Event.ARMED)                       # now in TAKEOFF (20 s limit)
    fake_now = fsm.entered_at + 21
    monkeypatch.setattr("fsm.mission_fsm.time.monotonic", lambda: fake_now)
    fsm.tick()
    assert fsm.state is State.RTL


def test_search_latches_reset_on_reentry():
    fsm = MissionFSM()
    reach_search(fsm)
    fsm.step(Event.IN_ZONE)
    fsm.step(Event.QR_MATCH)
    assert fsm.qr_matched and fsm.zone_ok
    fsm._goto(State.SEARCH_QR, reason="test")   # simulate re-entering search
    assert not fsm.qr_matched and not fsm.zone_ok


def test_ignored_events_do_not_change_state():
    fsm = MissionFSM()
    fsm.step(Event.ARMED)
    before = fsm.state
    fsm.step(Event.BANNER_FOUND)                # not valid in TAKEOFF
    assert fsm.state is before


def test_terminal_state_is_sticky():
    fsm = MissionFSM()
    fsm.step(Event.ARMED)
    fsm.step(Event.LINK_LOST)                   # -> RTL
    fsm.step(Event.ARMED)
    assert fsm.state is State.RTL