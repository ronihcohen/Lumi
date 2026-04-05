# Lumi UI

## Location

The UI is deployed and served from **`rony@tardis`**.

- **URL:** `http://tardis:8000`
- **Server script:** `~/code/Lumi/ui/server.py`
- **Files:** `~/code/Lumi/ui/` (includes `index.html`, `app.js`, `data/`)

## Running the server

```bash
ssh rony@tardis
cd ~/code/Lumi/ui
python3 server.py
```

## Deploying updates

From the local machine, sync changed files with:

```bash
rsync -av ui/server.py ui/app.js ui/index.html rony@tardis:~/code/Lumi/ui/
rsync -av ui/data/ rony@tardis:~/code/Lumi/ui/data/
```
