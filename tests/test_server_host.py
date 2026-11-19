"""The local server rejects requests whose Host header names another machine (DNS rebinding)."""

from ctb.server import _allowed_host


def test_host_header_check_blocks_dns_rebinding():
    assert _allowed_host("127.0.0.1:8800", "127.0.0.1")
    assert _allowed_host("localhost:8800", "127.0.0.1")
    assert _allowed_host("[::1]:8800", "127.0.0.1")
    assert not _allowed_host("evil.example:8800", "0.0.0.0")
    assert not _allowed_host("evil.example:8800", "127.0.0.1")
    assert _allowed_host("10.0.0.5:8800", "10.0.0.5")


def test_several_bind_addresses_are_all_accepted():
    bind = "127.0.0.1,100.96.108.128"
    assert _allowed_host("100.96.108.128:8800", bind)
    assert _allowed_host("127.0.0.1:8800", bind)
    assert not _allowed_host("100.64.0.9:8800", bind)
    assert not _allowed_host("evil.example:8800", bind)
