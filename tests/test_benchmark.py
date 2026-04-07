from benchmark import _console_print


def test_console_print_replaces_console_unsafe_symbols(capsys):
    _console_print("✓ ok ± std × size ✗ fail")

    captured = capsys.readouterr()

    assert "[OK] ok +/- std x size [FAIL] fail" in captured.out
