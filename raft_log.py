#!/usr/bin/env python3
"""raft_log.py — Persistent Raft log with write-ahead journaling.

Append-only log with CRC checksums, compaction via snapshots,
binary serialization, and crash recovery. The storage layer that
backs a Raft consensus implementation.

One file. Zero deps. Does one thing well.
"""

import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass


MAGIC = b'RAFT'
VERSION = 1
ENTRY_HEADER = struct.Struct('!IQIH')  # term(u32), index(u64), data_len(u32), crc(u16)


@dataclass
class LogEntry:
    term: int
    index: int
    data: bytes
    crc: int = 0

    def compute_crc(self) -> int:
        h = hashlib.md5(struct.pack('!IQ', self.term, self.index) + self.data).digest()
        return struct.unpack('!H', h[:2])[0]

    def encode(self) -> bytes:
        self.crc = self.compute_crc()
        header = ENTRY_HEADER.pack(self.term, self.index, len(self.data), self.crc)
        return header + self.data

    @classmethod
    def decode(cls, data: bytes, offset: int = 0) -> tuple['LogEntry', int]:
        term, index, data_len, crc = ENTRY_HEADER.unpack_from(data, offset)
        offset += ENTRY_HEADER.size
        payload = data[offset:offset + data_len]
        entry = cls(term, index, payload, crc)
        if entry.compute_crc() != crc:
            raise ValueError(f"CRC mismatch at index {index}")
        return entry, offset + data_len


@dataclass
class Snapshot:
    last_index: int
    last_term: int
    data: bytes  # serialized state machine

    def encode(self) -> bytes:
        header = struct.pack('!QI', self.last_index, self.last_term)
        return MAGIC + struct.pack('!I', len(header) + len(self.data)) + header + self.data

    @classmethod
    def decode(cls, raw: bytes) -> 'Snapshot':
        assert raw[:4] == MAGIC
        total = struct.unpack('!I', raw[4:8])[0]
        last_index, last_term = struct.unpack('!QI', raw[8:20])
        data = raw[20:8 + total]
        return cls(last_index, last_term, data)


class RaftLog:
    """Persistent append-only Raft log with snapshots."""

    def __init__(self, path: str | None = None):
        self.entries: list[LogEntry] = []
        self.snapshot: Snapshot | None = None
        self.path = path
        self.committed = 0
        if path and os.path.exists(path):
            self._recover()

    @property
    def first_index(self) -> int:
        if self.snapshot:
            return self.snapshot.last_index + 1
        return self.entries[0].index if self.entries else 1

    @property
    def last_index(self) -> int:
        if self.entries:
            return self.entries[-1].index
        if self.snapshot:
            return self.snapshot.last_index
        return 0

    @property
    def last_term(self) -> int:
        if self.entries:
            return self.entries[-1].term
        if self.snapshot:
            return self.snapshot.last_term
        return 0

    def append(self, term: int, data: bytes) -> LogEntry:
        index = self.last_index + 1
        entry = LogEntry(term, index, data)
        self.entries.append(entry)
        if self.path:
            self._persist_entry(entry)
        return entry

    def get(self, index: int) -> LogEntry | None:
        for e in self.entries:
            if e.index == index:
                return e
        return None

    def get_range(self, start: int, end: int) -> list[LogEntry]:
        return [e for e in self.entries if start <= e.index < end]

    def truncate_after(self, index: int):
        """Remove all entries after index (for log repair)."""
        self.entries = [e for e in self.entries if e.index <= index]
        if self.path:
            self._rewrite()

    def commit(self, index: int):
        self.committed = min(index, self.last_index)

    def take_snapshot(self, state_data: bytes):
        """Compact log by creating a snapshot up to committed index."""
        if self.committed == 0:
            return
        committed_entry = self.get(self.committed)
        if not committed_entry:
            return
        self.snapshot = Snapshot(self.committed, committed_entry.term, state_data)
        self.entries = [e for e in self.entries if e.index > self.committed]
        if self.path:
            snap_path = self.path + '.snap'
            with open(snap_path, 'wb') as f:
                f.write(self.snapshot.encode())
            self._rewrite()

    def _persist_entry(self, entry: LogEntry):
        mode = 'ab' if os.path.exists(self.path) else 'wb'
        with open(self.path, mode) as f:
            if mode == 'wb':
                f.write(MAGIC + struct.pack('!B', VERSION))
            f.write(entry.encode())

    def _rewrite(self):
        with open(self.path, 'wb') as f:
            f.write(MAGIC + struct.pack('!B', VERSION))
            for entry in self.entries:
                f.write(entry.encode())

    def _recover(self):
        # Load snapshot
        snap_path = self.path + '.snap'
        if os.path.exists(snap_path):
            with open(snap_path, 'rb') as f:
                self.snapshot = Snapshot.decode(f.read())

        # Load log entries
        if not os.path.exists(self.path):
            return
        with open(self.path, 'rb') as f:
            data = f.read()
        if len(data) < 5 or data[:4] != MAGIC:
            return
        offset = 5  # skip magic + version
        while offset < len(data):
            try:
                entry, offset = LogEntry.decode(data, offset)
                self.entries.append(entry)
            except (struct.error, ValueError):
                break  # Truncated/corrupt — stop here (crash recovery)

    def stats(self) -> dict:
        return {
            'entries': len(self.entries),
            'first': self.first_index,
            'last': self.last_index,
            'committed': self.committed,
            'has_snapshot': self.snapshot is not None,
            'snapshot_index': self.snapshot.last_index if self.snapshot else 0,
        }


def demo():
    print("=== Persistent Raft Log ===\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'raft.log')
        log = RaftLog(path)

        # Append entries
        for i in range(10):
            term = 1 if i < 5 else 2
            log.append(term, json.dumps({"op": "set", "key": f"k{i}", "val": i}).encode())
        log.commit(8)
        print(f"After 10 appends: {log.stats()}")

        # Snapshot
        state = json.dumps({"k0": 0, "k1": 1, "k2": 2}).encode()
        log.take_snapshot(state)
        print(f"After snapshot:   {log.stats()}")

        # Crash recovery
        log2 = RaftLog(path)
        print(f"After recovery:   {log2.stats()}")
        print(f"  Snapshot data: {log2.snapshot.data[:50] if log2.snapshot else 'none'}")

        # Verify entries survived
        for e in log2.entries:
            print(f"  [{e.term}:{e.index}] {e.data.decode()[:40]}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'test.log')
            log = RaftLog(path)
            # Append
            e1 = log.append(1, b'cmd1')
            e2 = log.append(1, b'cmd2')
            e3 = log.append(2, b'cmd3')
            assert log.last_index == 3
            assert log.get(2).data == b'cmd2'
            # CRC
            assert e1.crc == e1.compute_crc()
            # Range
            assert len(log.get_range(1, 3)) == 2
            # Truncate
            log.truncate_after(2)
            assert log.last_index == 2
            # Commit + snapshot
            log.append(2, b'cmd3_new')
            log.commit(2)
            log.take_snapshot(b'state_at_2')
            assert log.snapshot.last_index == 2
            assert len(log.entries) == 1  # only cmd3_new
            # Recovery
            log2 = RaftLog(path)
            assert log2.snapshot.last_index == 2
            assert log2.snapshot.data == b'state_at_2'
            assert len(log2.entries) == 1
            assert log2.entries[0].data == b'cmd3_new'
            # Corrupt entry detection — manually corrupt encoded bytes
            good = LogEntry(1, 1, b'hello')
            raw = bytearray(good.encode())
            raw[-1] ^= 0xFF  # flip last data byte
            try:
                LogEntry.decode(bytes(raw))
                assert False, "Should have detected corruption"
            except ValueError:
                pass
            print("All tests passed ✓")
    else:
        demo()
