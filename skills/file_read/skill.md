# file_read skill

## Purpose
Read local files and route by file type.

## Supported types
- PDF
- DOC
- DOCX
- TXT
- PNG
- JPG / JPEG

## Behavior
- Detect file type from suffix and basic content handling.
- For PDF / DOC / DOCX / TXT, extract text.
- For PNG / JPG, prepare image inputs for OCR or vision analysis.
- Return a structured document payload for downstream report analysis.

## Output
Return structured fields such as:
- doc_type
- path
- text
- chunks
- metadata
