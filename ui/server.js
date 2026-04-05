const express = require('express');
const path = require('path');
const fs = require('fs');

const PORT = process.env.PORT || 8000;
const DATA_DIR = 'data';
const STATIC_DIR = __dirname;
const SETTINGS_FILE = path.join(STATIC_DIR, 'settings.json');

function readSettings() {
  try {
    return JSON.parse(fs.readFileSync(SETTINGS_FILE, 'utf8'));
  } catch {
    return {};
  }
}

function writeSettings(settings) {
  fs.writeFileSync(SETTINGS_FILE, JSON.stringify(settings, null, 2));
}

const app = express();

app.use(express.json());

app.get('/api/books', (req, res) => {
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

app.get('/api/settings', (req, res) => {
  res.json(readSettings());
});

app.get('/api/settings/:key', (req, res) => {
  const settings = readSettings();
  res.json({ value: settings[req.params.key] ?? null });
});

app.put('/api/settings/:key', (req, res) => {
  try {
    const settings = readSettings();
    settings[req.params.key] = req.body.value;
    writeSettings(settings);
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

app.listen(PORT, () => {
  console.log(`Serving at http://localhost:${PORT}`);
}).on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    const altPort = 8001;
    app.listen(altPort, () => {
      console.log(`Port ${PORT} in use, serving at http://localhost:${altPort}`);
    }).on('error', (err) => {
      console.error(`Failed to start on fallback port ${altPort}:`, err);
      process.exit(1);
    });
  } else {
    throw err;
  }
});
