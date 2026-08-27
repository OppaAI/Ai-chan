from interface.webui.studio.log.backend.api import parse_log_records


def test_parse_log_records_groups_traceback_lines():
    records = parse_log_records(
        "2026-08-27 12:34:56.123  [ERROR   ]  system.demo — Broken\n"
        "Traceback (most recent call last):\n"
        "  File 'demo.py', line 1\n"
        "2026-08-27 12:34:57.456  [INFO    ]  system.demo — Recovered\n"
    )
    assert len(records) == 2
    assert records[0]["level"] == "ERROR"
    assert "Traceback" in records[0]["message"]
    assert records[1]["message"] == "Recovered"
