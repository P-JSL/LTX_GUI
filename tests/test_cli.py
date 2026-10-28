from ltx_gui.cli import main


def test_cli_rejects_missing_source(tmp_path):
    assert main([str(tmp_path / 'missing.mp4'), '--rectangle', '0,0,.2,.2']) == 1


def test_cli_bad_coordinates():
    # parse error exits with standard argparse status.
    import pytest
    with pytest.raises(SystemExit) as exc:
        main(['missing.mp4', '--rectangle', '0,0,0,0'])
    assert exc.value.code == 2
