"""Regression tests for the sandbox result channel (fd-3 wiring).

The child writes its structured JSON result to the pipe write-end inherited via
subprocess `pass_fds`. That fd is NOT remapped to fd 3, so the child must learn
its real number from SANDBOX_RESULT_FD. A regression here is silent: stdout /
figures / error vanish and the call still "succeeds" with empty output.
"""


from services.sandbox import run_python


def test_stdout_flows_through_result_channel():
    r = run_python('print("hello-from-user")', df=None, timeout=10)
    assert r.error is None
    assert r.exit_code == 0
    assert r.stdout == "hello-from-user\n"


def test_stderr_flows_through_result_channel():
    r = run_python('import sys; print("oops", file=sys.stderr)', df=None, timeout=10)
    assert r.error is None
    assert "oops" in r.stderr


def test_user_exception_reported_via_error_field():
    r = run_python('raise ValueError("boom")', df=None, timeout=10)
    assert r.error is not None
    assert "ValueError: boom" in r.error


def test_blocked_builtin_open_is_unavailable():
    # open() must not be reachable in user globals (sandbox escape guard).
    r = run_python('open("/tmp/should_not_exist", "w")', df=None, timeout=10)
    assert r.error is not None
    assert "NameError" in r.error
