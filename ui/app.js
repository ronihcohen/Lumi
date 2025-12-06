
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

    // Font size controls
    const increaseFontBtn = document.getElementById('increase-font');
    const decreaseFontBtn = document.getElementById('decrease-font');
    const FONT_SIZE_KEY = 'lumi_font_size';
    const MIN_FONT_SIZE = 12;
    const MAX_FONT_SIZE = 48;

    let currentFontSize = parseInt(localStorage.getItem(FONT_SIZE_KEY)) || 18;
    applyFontSize(currentFontSize);

    increaseFontBtn.addEventListener('click', () => {
        if (currentFontSize < MAX_FONT_SIZE) {
            currentFontSize += 2;
            applyFontSize(currentFontSize);
            localStorage.setItem(FONT_SIZE_KEY, currentFontSize);
        }
    });

    decreaseFontBtn.addEventListener('click', () => {
        if (currentFontSize > MIN_FONT_SIZE) {
            currentFontSize -= 2;
            applyFontSize(currentFontSize);
            localStorage.setItem(FONT_SIZE_KEY, currentFontSize);
        }
    });

    function applyFontSize(size) {
        bookContent.style.fontSize = size + 'px';
    }

    // Load available books
    try {
        const booksResponse = await fetch('/api/books');
        const books = await booksResponse.json();

        bookSelector.innerHTML = '';
        if (books.length === 0) {
            bookSelector.innerHTML = '<option value="">No books found</option>';
            bookContent.innerHTML = '<p style="color: red">No books found in data folder</p>';
            return;
        }

        // Get last selected book or use first one
        const lastBook = localStorage.getItem('lumi_selected_book');
        let selectedBook = books.includes(lastBook) ? lastBook : books[0];

        books.forEach(book => {
            const option = document.createElement('option');
            option.value = book;
            option.textContent = book;
            option.selected = book === selectedBook;
            bookSelector.appendChild(option);
        });

        // Handle book selection change
        bookSelector.addEventListener('change', () => {
            const newBook = bookSelector.value;
            if (newBook && newBook !== currentBook) {
                // Save current position before switching (regardless of playback state)
                if (currentBook) {
                    const oldStorageKey = `lumi_${currentBook}_last_time`;
                    localStorage.setItem(oldStorageKey, audioPlayer.currentTime);
                    console.log(`Saved position for ${currentBook}:`, audioPlayer.currentTime);
                }

                localStorage.setItem('lumi_selected_book', newBook);
                loadBook(newBook);
            }
        });

        // Load initial book
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
            cleanupListeners(); // Cleanup old listeners BEFORE changing source/loading

            console.log("Loading book:", bookName);
            currentBook = bookName;

            // Update audio source
            audioSource.src = `data/${encodeURIComponent(bookName)}.mp3`;
            audioPlayer.load();

            // Fetch book data
            const [wordsResponse, syncResponse] = await Promise.all([
                fetch(`data/${encodeURIComponent(bookName)}_transcribed_words.json`),
                fetch(`data/${encodeURIComponent(bookName)}_sync_map.json`)
            ]);

            wordsData = await wordsResponse.json();
            syncMap = await syncResponse.json();
            console.log("Data loaded. Words:", wordsData.length);

            // Pre-process paragraphs
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

            // Setup sync with book-specific storage key
            setupSync();

            // Event Delegation for word clicks
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

            while (wordIdx < wordsData.length) {
                const wordObj = wordsData[wordIdx];
                if (wordObj.start >= p.end) break;

                if (wordObj.start >= p.start) {
                    const wordSpan = document.createElement('span');
                    wordSpan.className = 'word';
                    wordSpan.textContent = wordObj.word;
                    wordSpan.setAttribute('data-start', wordObj.start);
                    wordSpan.setAttribute('data-end', wordObj.end);
                    pElement.appendChild(wordSpan);
                }
                wordIdx++;
            }
            fragment.appendChild(pElement);
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
        // Book-specific storage key
        const STORAGE_KEY = `lumi_${currentBook}_last_time`;
        let lastSaveTime = 0;

        const savedTime = parseFloat(localStorage.getItem(STORAGE_KEY));
        let initialTime = (!isNaN(savedTime) && isFinite(savedTime)) ? savedTime : 0;

        console.log(`Restoring position for ${currentBook}:`, initialTime);

        let pIdx = findClosestParagraphIndex(initialTime);
        let cIdx = Math.floor(pIdx / CHUNK_SIZE);
        renderChunk(cIdx);

        // Restore playback position
        if (initialTime > 0) {
            const restorePosition = () => {
                audioPlayer.currentTime = initialTime;
                console.log(`Position set to:`, audioPlayer.currentTime);
            };

            if (audioPlayer.readyState >= 2) {
                // Audio is already loaded enough to seek
                restorePosition();
            } else {
                // Wait until we can seek
                audioPlayer.addEventListener('canplay', restorePosition, { once: true });
            }
        }

        function updateHighlight() {
            const currentTime = audioPlayer.currentTime;

            const now = Date.now();
            if (now - lastSaveTime > 1000) {
                localStorage.setItem(STORAGE_KEY, currentTime);
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
            localStorage.setItem(STORAGE_KEY, audioPlayer.currentTime);
        }

        // Store references for next cleanup
        currentUpdateHighlight = updateHighlight;
        currentSavePausePosition = savePausePosition;

        // Add new event listeners
        audioPlayer.addEventListener('timeupdate', updateHighlight);
        audioPlayer.addEventListener('seeked', updateHighlight);
        audioPlayer.addEventListener('pause', savePausePosition);
    }
});
