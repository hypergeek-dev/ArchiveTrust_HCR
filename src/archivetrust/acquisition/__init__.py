"""Acquisition (ROADMAP.md §5.13.2, Revision 5; Constitution Article 25) — infrastructure only, a
sibling package to `runtime/` and `workspace/`.

Acquisition is the discover → validate → register process by which files become immutable Archive
Objects inside a Workspace. It never modifies files — only discovers, hashes, deduplicates, copies
into a Workspace's `archive/`, and registers. It may call the Trust Engine (`application.pipeline`)
once an Archive Object is registered, but the Trust Engine never calls back: no module in `domain`,
`providers`, `application`, `infrastructure`, or `runtime` may import this package (Article 25).
"""
