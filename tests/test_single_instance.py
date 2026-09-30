"""
Unit Tests for Single Instance Enforcement.
"""

from pathlib import Path
import fcntl
import os
import pytest


def test_single_instance_lock(tmp_path: Path):
    lock_file1 = tmp_path / "test.lock"
    f1 = open(lock_file1, "w")
    # First instance acquires lock successfully
    fcntl.flock(f1.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    # Second instance attempts to acquire same lock
    f2 = open(lock_file1, "w")
    with pytest.raises((IOError, OSError)):
        fcntl.flock(f2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    # Release first instance
    fcntl.flock(f1.fileno(), fcntl.LOCK_UN)
    f1.close()

    # Second instance can now acquire lock
    fcntl.flock(f2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(f2.fileno(), fcntl.LOCK_UN)
    f2.close()
