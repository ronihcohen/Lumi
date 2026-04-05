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
ssh rony@tardis
cd ~/code/Lumi/ui
git pull
```

### Start/Restart Server

```bash
ssh rony@tardis
cd ~/code/Lumi/ui
pm2 restart [NAME]
```