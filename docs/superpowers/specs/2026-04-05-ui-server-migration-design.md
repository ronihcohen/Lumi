# UI Server Migration - Node.js Implementation

## Overview
Convert Python UI server (server.py) to Node.js (server.js) with pm2 process management.

## Architecture

### Components
- `ui/server.js` - Express server replacing server.py

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/books` | GET | List available .mp3 books from data/ |
| `/api/settings` | GET | Get all settings from PostgreSQL |
| `/api/settings/:key` | GET | Get specific setting |
| `/api/settings/:key` | PUT | Update specific setting |
| `/*` | GET | Serve static files |

### Data Flow
1. Client requests `/api/books`
2. Server scans `ui/data/` for .mp3 files
3. Verifies corresponding `_transcribed_words.json` and `_sync_map.json` exist
4. Returns sorted list of valid books

### Database
- PostgreSQL via `pg` library
- DATABASE_URL from environment variable
- Default: `postgresql://admin:tombalulul777@localhost:5432/my_database`

## Implementation

### Dependencies
- express - HTTP server
- pg - PostgreSQL client

### Static Files
- Directory: `ui/`
- Audio files: `data/*.mp3`
- JSON data: `data/*_transcribed_words.json`, `data/*_sync_map.json`

### Headers
- Range requests enabled for audio seeking
- `.mp3`, `.json`: Cache 1 year
- `.html`, `.js`, `.css`: Cache 1 hour

### Port
- Default: 8000
- Fallback: 8001 if 8000 in use

## Error Handling
- Connection errors: Return JSON error
- Missing files: Skip gracefully
- DB errors: Log and return error response

## Testing
- Verify endpoints return correct JSON
- Test static file serving
- Verify caching headers
- Test port fallback