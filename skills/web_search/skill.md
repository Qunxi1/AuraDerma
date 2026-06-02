# web_search skill

## Purpose
Use Tavily to search the web when internal knowledge is insufficient.

## Inputs
- query: the user question or fallback search query
- top_k: number of results to return

## Behavior
- Call the Tavily-backed web search client.
- Return a small set of structured results.
- Keep results explicitly reference-only.
- The final answer must mark them as "网页搜索参考，仅供参考".

## Output
Return structured results with title, url, snippet, and source_note.
