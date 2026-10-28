from kernel_agent.common import ROOT, digest, read_json


def test_vendor_matches_recorded_hashes():
    manifest = read_json(ROOT / "PROVENANCE.json")
    for name, sha in manifest["vendor_sha256"].items():
        assert digest(ROOT / name) == sha, f"vendored file changed: {name}"
