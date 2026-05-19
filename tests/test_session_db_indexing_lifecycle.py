import sqlite3

from hermes_state import SessionDB


def test_sessiondb_rebuilds_missing_fts_rows(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("20260519_010101_recent", "cli")
    db.append_message("20260519_010101_recent", "user", "needle recent session content")
    db.close()

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM messages_fts")
    conn.execute("DELETE FROM messages_fts_trigram")
    conn.execute("UPDATE schema_version SET version = 12")
    conn.commit()
    conn.close()

    db = SessionDB(db_path)
    results = db.search_messages("needle", limit=5)
    assert [r["session_id"] for r in results] == ["20260519_010101_recent"]
    db.close()


def test_append_message_requires_existing_session(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.append_message("missing_session", "user", "should not create or orphan")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("append_message unexpectedly allowed an orphan message")
    assert db.search_sessions() == []
    db.close()
