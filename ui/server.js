const express = require('express');
const { Pool } = require('pg');
const path = require('path');
const fs = require('fs');

const PORT = process.env.PORT || 8000;
const DATA_DIR = 'data';
const STATIC_DIR = __dirname;

const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://admin:tombalulul777@localhost:5432/my_database';

const pool = new Pool({ connectionString: DATABASE_URL });

pool.query(`
  CREATE TABLE IF NOT EXISTS user_settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )
`).catch(e => console.error('Failed to initialize user_settings table:', e));

const app = express();

app.use(express.json());

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
    }).on('error', (err) => {
      console.error(`Failed to start on fallback port ${altPort}:`, err);
      process.exit(1);
    });
  } else {
    throw err;
  }
});
