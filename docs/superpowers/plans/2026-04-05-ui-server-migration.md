# UI Server Migration - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Python UI server to Node.js with Express, replace server.py with server.js

**Architecture:** Express server with PostgreSQL via pg library, static file serving, same API endpoints as Python version

**Tech Stack:** Node.js, Express, pg

---

### Task 1: Create package.json

**Files:**
- Create: `ui/package.json`

- [ ] **Step 1: Create package.json with dependencies**

```json
{
  "name": "lumi-ui",
  "version": "1.0.0",
  "description": "Lumi UI Server",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "pg": "^8.11.3"
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add ui/package.json
git commit -m "feat: add Node.js package.json for UI server"
```

---

### Task 2: Create server.js

**Files:**
- Create: `ui/server.js`

- [ ] **Step 1: Write server.js with Express and all endpoints**

```javascript
const express = require('express');
const { Pool } = require('pg');
const path = require('path');
const fs = require('fs');

const PORT = process.env.PORT || 8000;
const DATA_DIR = 'data';
const STATIC_DIR = __dirname;

const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://admin:tombalul777@localhost:5432/my_database';

const pool = new Pool({ connectionString: DATABASE_URL });

const app = express();

app.get('/api/books', async (req, res) => {
  try {
    const dataPath = path.join(STATIC_DIR, DATA_DIR);
    if (!fs.existsSync(dataPath)) {
      return res.json([]);
    }
    const files = fs.readdirSync(dataPath);
    const books = [];
    for (const file of files) {
      if (file.endsWith('.mp3')) {
        const baseName = file.slice(0, -4);
        const wordsFile = `${baseName}_transcribed_words.json`;
        const syncFile = `${baseName}_sync_map.json`;
        if (fs.existsSync(path.join(dataPath, wordsFile)) && 
            fs.existsSync(path.join(dataPath, syncFile))) {
          books.push(baseName);
        }
      }
    }
    books.sort();
    res.json(books);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/settings', async (req, res) => {
  try {
    const result = await pool.query('SELECT key, value FROM user_settings');
    const settings = {};
    for (const row of result.rows) {
      settings[row.key] = row.value;
    }
    res.json(settings);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/settings/:key', async (req, res) => {
  try {
    const result = await pool.query('SELECT value FROM user_settings WHERE key = $1', [req.params.key]);
    res.json({ value: result.rows[0]?.value || null });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.put('/api/settings/:key', async (req, res) => {
  try {
    await pool.query(
      `INSERT INTO user_settings (key, value, updated_at)
       VALUES ($1, $2, CURRENT_TIMESTAMP)
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP`,
      [req.params.key, req.body.value]
    );
    res.json({ success: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.use(express.static(STATIC_DIR, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.mp3') || filePath.endsWith('.json')) {
      res.set('Cache-Control', 'public, max-age=31536000, immutable');
    } else if (filePath.endsWith('.html') || filePath.endsWith('.js') || filePath.endsWith('.css')) {
      res.set('Cache-Control', 'public, max-age=3600');
    }
  }
}));

const server = app.listen(PORT, () => {
  console.log(`Serving at http://localhost:${PORT}`);
  console.log('Range requests enabled for audio seeking');
}).on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    const altPort = 8001;
    app.listen(altPort, () => {
      console.log(`Port ${PORT} in use, trying ${altPort}`);
      console.log(`Serving at http://localhost:${altPort}`);
    });
  } else {
    throw err;
  }
});
```

- [ ] **Step 2: Commit**

```bash
git add ui/server.js
git commit -m "feat: add Node.js Express server replacing server.py"
```

---

### Task 3: Install dependencies and test locally

**Files:**
- Modify: `ui/` (install dependencies)

- [ ] **Step 1: Install dependencies**

```bash
cd ui && npm install
```

- [ ] **Step 2: Test server starts**

```bash
cd ui && node server.js
```
Expected: "Serving at http://localhost:8000"

- [ ] **Step 3: Test API endpoint**

```bash
curl http://localhost:8000/api/books
```
Expected: JSON array of books

- [ ] **Step 4: Commit dependencies**

```bash
git add ui/node_modules/ 2>/dev/null || true
git add ui/package-lock.json 2>/dev/null || true
git add ui/package.json
git commit -m "chore: install Node.js dependencies"
```

---

### Task 4: Deploy to tardis server

**Files:**
- Modify: Remote tardis server

- [ ] **Step 1: Push local changes to git**

```bash
git push
```

- [ ] **Step 2: SSH to tardis and pull changes**

```bash
ssh rony@tardis "cd ~/code/Lumi && git pull"
```

- [ ] **Step 3: Install dependencies on tardis**

```bash
ssh rony@tardis "cd ~/code/Lumi/ui && npm install"
```

- [ ] **Step 4: Check current pm2 processes**

```bash
ssh rony@tardis "pm2 list"
```

- [ ] **Step 5: Stop old Python server (if running)**

```bash
ssh rony@tardis "pm2 delete server.py 2>/dev/null || pm2 delete all"
```

- [ ] **Step 6: Start Node.js server with pm2**

```bash
ssh rony@tardis "cd ~/code/Lumi/ui && pm2 start server.js --name lumi-ui"
```

- [ ] **Step 7: Verify server is running**

```bash
ssh rony@tardis "pm2 list"
```

- [ ] **Step 8: Test remote server**

```bash
curl http://tardis:8000/api/books
```