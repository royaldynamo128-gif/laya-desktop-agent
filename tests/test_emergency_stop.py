"""
Unit Tests for Emergency Stop Latch and Actions.
"""

from safety.emergency_stop import EmergencyStop


def test_emergency_stop_lifecycle():
    estop = EmergencyStop()
    assert estop.is_stopped() is False

    callback_called = False
    def on_stop():
        nonlocal callback_called
        callback_called = True

    estop.register_callback(on_stop)

    # Trigger emergency stop
    estop.trigger("User initiated test abort")
    assert estop.is_stopped() is True
    assert callback_called is True

    # Reset
    estop.reset()
    assert estop.is_stopped() is False
