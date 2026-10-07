"""Regenerate the current single deliverable notebook."""
from build_diego_deliverable import main
from build_evidence_manifest import main as build_manifest

if __name__ == '__main__':
    main()
    build_manifest()
