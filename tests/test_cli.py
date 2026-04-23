from satbba.cli.main import main


def test_list_matchers_command(capsys) -> None:
    code = main(["list-matchers"])
    captured = capsys.readouterr()
    assert code == 0
    assert "sift" in captured.out
