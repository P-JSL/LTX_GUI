from ltx_gui.telemetry import parse_nvidia_smi


def test_parse_multiple_gpus_and_errors():
    results = parse_nvidia_smi('0, NVIDIA RTX 4090, 24564, 6500, 87\n'
                               '1, NVIDIA RTX 3090, 24576, 600, 12\n'
                               'bad, card, N/A, N/A, N/A\n')
    assert len(results) == 2
    assert results[0].index == 0
    assert results[0].utilization_percent == 87
    assert results[1].used_mib == 600
