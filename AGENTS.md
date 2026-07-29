# ArchiveTrust Codex Project Instructions

## ChatGPT oversight context pack

Maintain `For Chat-GPT project folder/` as the user's upload-ready context pack for a ChatGPT
Project that oversees ArchiveTrust.

- Keep the pack flat: files only, with no subfolders.
- Keep the pack at or below the ChatGPT Project limit of 25 files.
- Preserve the source document filenames unless a rename is necessary to avoid a collision.
- Keep `PROJECT_OVERSIGHT_INDEX.md` current with the reading order, authority precedence, source
  mapping, snapshot date, and pack file count.
- When an included repository source document changes, refresh its corresponding copy in the pack.
- When important project understanding changes, decide whether to update an existing pack document
  or add another independent Markdown file. Prefer updating or consolidating before adding files.
- Do not add transient workstream notes when their conclusions are already represented by an
  authoritative current document or consolidated report.
- After changing the pack, verify that it contains no subfolders, contains no more than 25 files,
  and that copied source documents match their repository originals.
- In the final response for any task that changes the pack, explicitly tell the user that the
  ChatGPT Project context pack was updated and list the files they need to replace or upload. The
  user updates the external ChatGPT Project manually.
- Keep the pack directory Git-ignored.
