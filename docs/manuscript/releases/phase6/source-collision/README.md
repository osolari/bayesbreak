# Source Archive Case Collision

The signed unified-source ZIP contains two different files whose names differ only by
case:

- `shared/bibliography/ANNOTATION_MANIFEST.json`
- `shared/bibliography/annotation_manifest.json`

They cannot coexist in one directory on the default case-insensitive macOS filesystem.
All Phase 4R, Phase 5, and Phase 6 validators and the annotated-literature appendix use
the lowercase `annotation_manifest.json`, which is therefore retained at its canonical
source path.

The uppercase key-only list is preserved byte-for-byte in this directory with SHA-256
`8d1cb33de3cd3f940e239f566b63ff62dec516a6d0da3c5507032439488b6b1e`.
The canonical lowercase registry has SHA-256
`6a34aca89a1ce05b297ef8321327c74678f03b24e4b58ef4ada7c7fa332a539e`.
The original signed nested ZIP is also retained in the Phase 6 release directory.
