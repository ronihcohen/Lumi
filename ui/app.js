
document.addEventListener('DOMContentLoaded', async () => {
    const audioPlayer = document.getElementById('audio-player');
    const bookContent = document.getElementById('book-content');

    let wordsData = [];
    let syncMap = {};
    let sortedParagraphs = []; // Array of {id, start, end}
    let currentChunkIndex = -1;
    const CHUNK_SIZE = 50; // Number of paragraphs per chunk

    // Force audio to start loading
    audioPlayer.load();

    // Font size controls
    const increaseFontBtn = document.getElementById('increase-font');
    const decreaseFontBtn = document.getElementById('decrease-font');
    const FONT_SIZE_KEY = 'lumi_font_size';
    const MIN_FONT_SIZE = 12;
    const MAX_FONT_SIZE = 48;

    // Load saved font size
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

    try {
        console.log("Fetching data...");
        const [wordsResponse, syncResponse] = await Promise.all([
            fetch('data/The Fellowship of the Ring Lord of the Rings, Book 1_transcribed_words.json'),
            fetch('data/The Fellowship of the Ring Lord of the Rings, Book 1_sync_map.json')
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

        // Initial setup
        setupSync();

        // Event Delegation for click (seek)
        bookContent.addEventListener('click', (e) => {
            if (e.target.classList.contains('word')) {
                const rawStart = e.target.getAttribute('data-start');
                const start = parseFloat(rawStart);

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
        });

    } catch (error) {
        console.error('Error loading data:', error);
        bookContent.innerHTML = `<p style="color: red">Error loading book data: ${error.message}</p>`;
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

        // Reapply font size after rendering
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
        const STORAGE_KEY = 'lumi_last_time';
        let lastSaveTime = 0;

        // Resume & Initial Render
        const savedTime = parseFloat(localStorage.getItem(STORAGE_KEY));
        let initialTime = (!isNaN(savedTime) && isFinite(savedTime)) ? savedTime : 0;

        let pIdx = findClosestParagraphIndex(initialTime);
        let cIdx = Math.floor(pIdx / CHUNK_SIZE);
        renderChunk(cIdx);

        if (initialTime > 0) {
            if (audioPlayer.readyState >= 1) audioPlayer.currentTime = initialTime;
            else audioPlayer.addEventListener('loadedmetadata', () => { audioPlayer.currentTime = initialTime; }, { once: true });
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

            // Search the DOM elements directly (not the full data array)
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

                // Always keep highlighted word centered
                activeSpan.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }

        audioPlayer.addEventListener('timeupdate', updateHighlight);
        audioPlayer.addEventListener('seeked', updateHighlight);
        audioPlayer.addEventListener('pause', () => localStorage.setItem(STORAGE_KEY, audioPlayer.currentTime));
    }
});
