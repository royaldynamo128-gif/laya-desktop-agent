"""
Unit tests for MethodRegistry and 7-tier ActionMethod priority handling.
"""

import pytest
from planning.method_registry import ActionMethod, MethodRegistry


def test_method_registry_open_application_hierarchy():
    reg = MethodRegistry()
    methods = reg.get_methods_for_action("open_application")
    assert methods[0] == ActionMethod.NATIVE_API
    assert ActionMethod.KDE_AUTOMATION in methods


def test_method_registry_typing_hierarchy():
    reg = MethodRegistry()
    methods = reg.get_methods_for_action("type_text")
    assert methods[0] == ActionMethod.KEYBOARD
    assert ActionMethod.ACCESSIBILITY in methods


def test_method_registry_next_method_progression():
    reg = MethodRegistry()
    failed = []

    m1 = reg.get_next_method("open_application", failed)
    assert m1 == ActionMethod.NATIVE_API
    failed.append(m1)

    m2 = reg.get_next_method("open_application", failed)
    assert m2 == ActionMethod.ACCESSIBILITY
    failed.append(m2)

    m3 = reg.get_next_method("open_application", failed)
    assert m3 == ActionMethod.KDE_AUTOMATION
    failed.append(m3)

    m4 = reg.get_next_method("open_application", failed)
    assert m4 == ActionMethod.KEYBOARD
    failed.append(m4)

    # All exhausted
    m5 = reg.get_next_method("open_application", failed)
    assert m5 is None
