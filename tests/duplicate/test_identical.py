"""Tests for deh.duplicate.identical byte-for-byte duplicate detection & survivor selection."""
from pathlib import Path

from deh.duplicate.identical import IdenticalFileResolver


def _w(root: Path, rel: str, data: bytes) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p.resolve()


class TestGroupDetection:
    def test_groups_only_true_byte_duplicates(self, tmp_path):
        a = _w(tmp_path, "x.jpg", b"AAAA")                    # size 4
        b = _w(tmp_path, "sub/y.jpg", b"AAAA")                # same bytes as a -> grouped
        c = _w(tmp_path, "z.png", b"BBBBB")                   # size 5, unique content
        d = _w(tmp_path, "same_size_diff.bin", b"CCCCC")      # size 5 like c but different bytes

        groups = IdenticalFileResolver(str(tmp_path)).group([a, b, c, d])

        assert len(groups) == 1
        assert {str(p) for p in groups[0]} == {str(a), str(b)}

    def test_no_group_for_unique_files(self, tmp_path):
        u = _w(tmp_path, "u.jpg", b"unique-one!")
        v = _w(tmp_path, "v.jpg", b"different-bytes-2")
        assert IdenticalFileResolver(str(tmp_path)).group([u, v]) == []

    def test_single_file_never_a_duplicate(self, tmp_path):
        only = _w(tmp_path, "only.jpg", b"data")
        assert IdenticalFileResolver(str(tmp_path)).group([only]) == []


class TestKeeperSelection:
    def test_deepest_copy_survives(self, tmp_path):
        data = b"identical-content"
        shallow = _w(tmp_path, "top.jpg", data)               # depth 0
        mid = _w(tmp_path, "d1/mid.jpg", data)                # depth 1
        deep = _w(tmp_path, "d1/d2/deep.jpg", data)           # depth 2 (deepest -> keeper)

        r = IdenticalFileResolver(str(tmp_path))
        group = r.group([shallow, mid, deep])[0]
        assert str(r.keeper_for(group)) == str(deep)

    def test_equal_depth_tie_breaks_to_smallest_path(self, tmp_path):
        data = b"tie-break-bytes"
        t1 = _w(tmp_path, "aa/x.jpg", data)                   # both depth 1
        t2 = _w(tmp_path, "bb/y.jpg", data)

        r = IdenticalFileResolver(str(tmp_path))
        expected = min(t1, t2, key=lambda p: str(p))
        assert str(r.keeper_for(r.group([t1, t2])[0])) == str(expected)


class TestResolvePartition:
    def test_partition_covers_only_duplicate_members(self, tmp_path):
        data = b"dup-dup-dup"
        a = _w(tmp_path, "a.jpg", data)                       # depth 0 -> redundant
        b = _w(tmp_path, "sub/b.jpg", data)                   # depth 1 -> redundant
        c = _w(tmp_path, "sub/deeper/c.jpg", data)            # depth 2 -> keeper (deepest)
        u = _w(tmp_path, "unique.png", b"not-a-duplicate")

        keepers, redundant = IdenticalFileResolver(str(tmp_path)).resolve([a, b, c, u])

        assert {str(p) for p in keepers} == {str(c)}
        assert {str(p) for p in redundant} == {str(a), str(b)}
        # The unique file appears in neither list.
        assert all(str(u) != str(p) for p in keepers + redundant)

    def test_no_duplicates_returns_empty_lists(self, tmp_path):
        u = _w(tmp_path, "u.jpg", b"one")
        v = _w(tmp_path, "v.jpg", b"two-two")
        keepers, redundant = IdenticalFileResolver(str(tmp_path)).resolve([u, v])
        assert keepers == [] and redundant == []


class TestDepth:
    def test_depth_of_relative_to_root(self, tmp_path):
        r = IdenticalFileResolver(str(tmp_path))
        top = _w(tmp_path, "top.jpg", b"x")
        nested = _w(tmp_path, "a/b/c/deep.jpg", b"x")
        assert r.depth_of(top) == 0
        assert r.depth_of(nested) == 3
