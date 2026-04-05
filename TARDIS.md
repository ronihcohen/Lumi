## 3. UI Deployment

Push processed files to the remote server.

### Server Details

- **Host:** `rony@tardis`
- **URL:** `http://tardis:8000`
- **Files location:** `~/code/Lumi/ui/`

### Deploy Data Files

```bash
rsync -av ui/data/ rony@tardis:~/code/Lumi/ui/data/
```

### Deploy UI Updates

```bash
rsync -av ui/server.py ui/app.js ui/index.html rony@tardis:~/code/Lumi/ui/
```

### Start/Restart Server

```bash
ssh rony@tardis
cd ~/code/Lumi/ui
pm2 restart [NAME]
```