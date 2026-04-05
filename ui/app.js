
document.addEventListener('DOMContentLoaded', async () => {
    const audioPlayer = document.getElementById('audio-player');
    const audioSource = document.getElementById('audio-source');
    const bookContent = document.getElementById('book-content');
    const bookSelector = document.getElementById('book-selector');

    let wordsData = [];
    let syncMap = {};
    let sortedParagraphs = [];
    let currentChunkIndex = -1;
    let currentBook = null;
    let currentUpdateHighlight = null;
    let currentSavePausePosition = null;
    const CHUNK_SIZE = 50;

    let serverSettings = {};
    let settingsLoaded = false;
    let pendingSaves = {};
    let syncTimeout = null;

    function loadSettings() {
        loadSettingsFromServer();
    }

    function loadSettingsFromServer() {
        fetch('/api/settings').then(resp => resp.json()).then(data => {
            serverSettings = data;
            for (const [key, value] of Object.entries(serverSettings)) {
                localStorage.setItem(key, value);
            }
            settingsLoaded = true;
        }).catch(e => {
            console.warn('Could not load settings from server:', e);
        });
    }

    async function saveSetting(key, value) {
        localStorage.setItem(key, value);
        serverSettings[key] = value;
        pendingSaves[key] = value;
        
        clearTimeout(syncTimeout);
        syncTimeout = setTimeout(async () => {
            const toSave = { ...pendingSaves };
            pendingSaves = {};
            for (const [k, v] of Object.entries(toSave)) {
                try {
                    await fetch(`/api/settings/${encodeURIComponent(k)}`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ value: v })
                    });
                } catch (e) {
                    console.warn('Could not save setting to server:', e);
                }
            }
        }, 3000);
    }

    function getSetting(key, defaultValue) {
        const localValue = localStorage.getItem(key);
        if (localValue !== null) {
            return localValue;
        }
        if (settingsLoaded && serverSettings.hasOwnProperty(key)) {
            return serverSettings[key];
        }
        return defaultValue;
    }

    const increaseFontBtn = document.getElementById('increase-font');
    const decreaseFontBtn = document.getElementById('decrease-font');
    const FONT_SIZE_KEY = 'lumi_font_size';
    const MIN_FONT_SIZE = 12;
    const MAX_FONT_SIZE = 48;

    let currentFontSize = parseInt(getSetting(FONT_SIZE_KEY, '18')) || 18;
    applyFontSize(currentFontSize);

    increaseFontBtn.addEventListener('click', () => {
        if (currentFontSize < MAX_FONT_SIZE) {
            currentFontSize += 2;
            applyFontSize(currentFontSize);
            saveSetting(FONT_SIZE_KEY, currentFontSize.toString());
        }
    });

    decreaseFontBtn.addEventListener('click', () => {
        if (currentFontSize > MIN_FONT_SIZE) {
            currentFontSize -= 2;
            applyFontSize(currentFontSize);
            saveSetting(FONT_SIZE_KEY, currentFontSize.toString());
        }
    });

    function applyFontSize(size) {
        bookContent.style.fontSize = size + 'px';
    }

    loadSettings();

    try {
        const booksResponse = await fetch('/api/books');
        const books = await booksResponse.json();

        bookSelector.innerHTML = '';
        if (books.length === 0) {
            bookSelector.innerHTML = '<option value="">No books found</option>';
            bookContent.innerHTML = '<p style="color: red">No books found in data folder</p>';
            return;
        }

        const lastBook = getSetting('lumi_selected_book', '');
        let selectedBook = books.includes(lastBook) ? lastBook : books[0];

        books.forEach(book => {
            const option = document.createElement('option');
            option.value = book;
            option.textContent = book;
            option.selected = book === selectedBook;
            bookSelector.appendChild(option);
        });

        bookSelector.addEventListener('change', () => {
            const newBook = bookSelector.value;
            if (newBook && newBook !== currentBook) {
                if (currentBook) {
                    const oldStorageKey = `lumi_${currentBook}_last_time`;
                    saveSetting(oldStorageKey, audioPlayer.currentTime.toString());
                    console.log(`Saved position for ${currentBook}:`, audioPlayer.currentTime);
                }

                saveSetting('lumi_selected_book', newBook);
                loadBook(newBook);
            }
        });

        await loadBook(selectedBook);

    } catch (error) {
        console.error('Error loading books:', error);
        bookContent.innerHTML = `<p style="color: red">Error loading books: ${error.message}</p>`;
    }

    function cleanupListeners() {
        if (currentUpdateHighlight) {
            audioPlayer.removeEventListener('timeupdate', currentUpdateHighlight);
            audioPlayer.removeEventListener('seeked', currentUpdateHighlight);
            currentUpdateHighlight = null;
        }
        if (currentSavePausePosition) {
            audioPlayer.removeEventListener('pause', currentSavePausePosition);
            currentSavePausePosition = null;
        }
    }

    async function loadBook(bookName) {
        try {
            cleanupListeners();

            console.log("Loading book:", bookName);
            currentBook = bookName;

            audioSource.src = `data/${encodeURIComponent(bookName)}.mp3`;
            audioPlayer.load();

            const [wordsResponse, syncResponse] = await Promise.all([
                fetch(`data/${encodeURIComponent(bookName)}_transcribed_words.json`),
                fetch(`data/${encodeURIComponent(bookName)}_sync_map.json`)
            ]);

            wordsData = await wordsResponse.json();
            syncMap = await syncResponse.json();
            console.log("Data loaded. Words:", wordsData.length);

            const pIds = Object.keys(syncMap).sort((a, b) => {
                const numA = parseInt(a.slice(1));
                const numB = parseInt(b.slice(1));
                return numA - numB;
            });

            sortedParagraphs = pIds.map(id => ({
                id: id,
                start: syncMap[id].start,
                end: syncMap[id].end
            }));

            setupSync();

            bookContent.removeEventListener('click', handleWordClick);
            bookContent.addEventListener('click', handleWordClick);

        } catch (error) {
            console.error('Error loading book:', error);
            bookContent.innerHTML = `<p style="color: red">Error loading book data: ${error.message}</p>`;
        }
    }

    function handleWordClick(e) {
        if (e.target.classList.contains('word')) {
            const start = parseFloat(e.target.getAttribute('data-start'));
            if (!isNaN(start)) {
                const doSeek = () => {
                    audioPlayer.currentTime = start;
                    audioPlayer.play().catch(err => console.error("Play failed:", err));
                };

                if (audioPlayer.seekable.length > 0) {
                    doSeek();
                } else {
                    audioPlayer.addEventListener('canplay', doSeek, { once: true });
                }
            }
        }
    }

    function renderChunk(chunkIndex) {
        if (chunkIndex < 0 || chunkIndex * CHUNK_SIZE >= sortedParagraphs.length) return;

        console.log(`Rendering chunk ${chunkIndex}`);
        bookContent.innerHTML = '';

        const startIndex = chunkIndex * CHUNK_SIZE;
        const endIndex = Math.min((chunkIndex + 1) * CHUNK_SIZE, sortedParagraphs.length);
        const chunkParagraphs = sortedParagraphs.slice(startIndex, endIndex);
        const fragment = document.createDocumentFragment();

        let wordIdx = findWordIndex(chunkParagraphs[0].start);

        chunkParagraphs.forEach(p => {
            const pElement = document.createElement('p');
            pElement.id = p.id;
            pElement.style.margin = '0';
            pElement.style.display = 'inline';

            let hasWords = false;
            while (wordIdx < wordsData.length) {
                const wordObj = wordsData[wordIdx];
                if (wordObj.start >= p.end) break;

                if (wordObj.start >= p.start) {
                    const wordText = wordObj.word.trim();
                    if (!wordText || wordText.length < 2) {
                        wordIdx++;
                        continue;
                    }
                    const wordSpan = document.createElement('span');
                    wordSpan.className = 'word';
                    wordSpan.textContent = wordText + ' ';
                    wordSpan.setAttribute('data-start', wordObj.start);
                    wordSpan.setAttribute('data-end', wordObj.end);
                    pElement.appendChild(wordSpan);
                    hasWords = true;
                }
                wordIdx++;
            }
            if (hasWords) {
                fragment.appendChild(pElement);
            }
        });

        bookContent.appendChild(fragment);
        currentChunkIndex = chunkIndex;
        applyFontSize(currentFontSize);
    }

    function findWordIndex(time) {
        let low = 0, high = wordsData.length - 1;
        while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            if (wordsData[mid].start < time) low = mid + 1;
            else high = mid - 1;
        }
        return low;
    }

    function findClosestParagraphIndex(time) {
        let low = 0, high = sortedParagraphs.length - 1;
        let best = 0;
        while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            if (sortedParagraphs[mid].start <= time) {
                best = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return best;
    }

    function setupSync() {
        const STORAGE_KEY = `lumi_${currentBook}_last_time`;
        let lastSaveTime = 0;

        const savedTime = parseFloat(getSetting(STORAGE_KEY, '0'));
        let initialTime = (!isNaN(savedTime) && isFinite(savedTime)) ? savedTime : 0;

        console.log(`Restoring position for ${currentBook}:`, initialTime);

        let pIdx = findClosestParagraphIndex(initialTime);
        let cIdx = Math.floor(pIdx / CHUNK_SIZE);
        renderChunk(cIdx);

        if (initialTime > 0) {
            const restorePosition = () => {
                audioPlayer.currentTime = initialTime;
                console.log(`Position set to:`, audioPlayer.currentTime);
            };

            if (audioPlayer.readyState >= 2) {
                restorePosition();
            } else {
                audioPlayer.addEventListener('canplay', restorePosition, { once: true });
            }
        }

        function updateHighlight() {
            const currentTime = audioPlayer.currentTime;

            const now = Date.now();
            if (now - lastSaveTime > 1000) {
                saveSetting(STORAGE_KEY, currentTime.toString());
                lastSaveTime = now;
            }

            let pIdx = findClosestParagraphIndex(currentTime);
            let targetChunk = Math.floor(pIdx / CHUNK_SIZE);
            if (targetChunk !== currentChunkIndex) {
                renderChunk(targetChunk);
            }

            const wordSpans = document.getElementsByClassName('word');
            let found = -1;
            let low = 0, high = wordSpans.length - 1;

            while (low <= high) {
                const mid = Math.floor((low + high) / 2);
                const span = wordSpans[mid];
                const start = parseFloat(span.getAttribute('data-start'));
                const end = parseFloat(span.getAttribute('data-end'));

                if (currentTime >= start && currentTime < end) {
                    found = mid;
                    break;
                } else if (currentTime < start) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            }

            if (found !== -1) {
                const prev = document.querySelector('.active-word');
                if (prev) prev.classList.remove('active-word');

                const activeSpan = wordSpans[found];
                activeSpan.classList.add('active-word');
                activeSpan.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }

        function savePausePosition() {
            saveSetting(STORAGE_KEY, audioPlayer.currentTime.toString());
        }

        currentUpdateHighlight = updateHighlight;
        currentSavePausePosition = savePausePosition;

        audioPlayer.addEventListener('timeupdate', updateHighlight);
        audioPlayer.addEventListener('seeked', updateHighlight);
        audioPlayer.addEventListener('pause', savePausePosition);
    }
});
