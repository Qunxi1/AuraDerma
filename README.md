# AuraDerma

## Quick start

1. Copy the example config:

```powershell
Copy-Item .env.example .env
```

2. Fill in your real API keys in `.env`.

3. Install dependencies:

```bash
pip install -e .
```

4. Start Qdrant:

```bash
docker compose up -d qdrant
```

5. Run the CLI:

```bash
auraderma chat
```

## Config files

- `.env.example` is the template
- `.env` is your real local configuration
- `.env.local` can be used for machine-specific overrides

## Skills

The only fully wired skill right now is `web_search`, backed by Tavily.
