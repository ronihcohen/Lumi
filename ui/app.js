

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
                console.log("Clicked word. Raw data-start:", rawStart);
                const start = parseFloat(rawStart);

                if (!isNaN(start)) {
                    console.log("Seeking to:", start);
                    console.log("Audio seekable range:", audioPlayer.seekable.length > 0 ? audioPlayer.seekable.end(0) : "None");
                    console.log("Current Time before seek:", audioPlayer.currentTime);

                    const doSeek = () => {
                        console.log("Executing seek to:", start);
                        audioPlayer.currentTime = start;
                        audioPlayer.play().catch(err => console.error("Play failed:", err));
                    };

                    if (audioPlayer.seekable.length > 0) {
                        doSeek();
                    } else {
                        console.log("Audio not seekable yet, waiting for canplay...");
                        audioPlayer.addEventListener('canplay', () => {
                            console.log("Audio ready, now seeking");
                            doSeek();
                        }, { once: true });
                    }
                } else {
                    console.error("Invalid start time parsed:", start);
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
        bookContent.innerHTML = ''; // Clear previous chunk

        const startIndex = chunkIndex * CHUNK_SIZE;
        const endIndex = Math.min((chunkIndex + 1) * CHUNK_SIZE, sortedParagraphs.length);

        const chunkParagraphs = sortedParagraphs.slice(startIndex, endIndex);
        const fragment = document.createDocumentFragment();

        // We need words that fall within this chunk's time range
        const chunkStartTime = chunkParagraphs[0].start;
        const chunkEndTime = chunkParagraphs[chunkParagraphs.length - 1].end;

        // Find start word index
        let wordIdx = findWordIndex(chunkStartTime);

        chunkParagraphs.forEach(p => {
            const pElement = document.createElement('p');
            pElement.id = p.id;

            while (wordIdx < wordsData.length) {
                const wordObj = wordsData[wordIdx];
                if (wordObj.start >= p.end) break; // Belongs to next paragraph

                if (wordObj.start >= p.start) { // Should be true if sorted correctly
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
        // Scroll to top of content when changing chunks manually? 
        // Or let sync handle it.
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

    function findParagraphIndex(time) {
        let low = 0, high = sortedParagraphs.length - 1;
        let idx = -1;
        while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            if (time >= sortedParagraphs[mid].start && time <= sortedParagraphs[mid].end) {
                return mid;
            }
            if (time < sortedParagraphs[mid].start) high = mid - 1;
            else low = mid + 1;
        }
        // Fallback: mostly likely in a gap or silence. Return closest previous?
        // If low > 0, return low - 1 (the paragraph that ended before this time).
        // Or simple constraint:
        if (arrCheck(low)) return low;
        if (arrCheck(high)) return high;
        return -1;

        function arrCheck(i) {
            return i >= 0 && i < sortedParagraphs.length && time >= sortedParagraphs[i].start && time <= sortedParagraphs[i].end;
        }
    }

    // Rough search for "closest future paragraph" if silence
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

        // Render initial state
        let pIdx = findClosestParagraphIndex(initialTime);
        let cIdx = Math.floor(pIdx / CHUNK_SIZE);
        renderChunk(cIdx);

        if (initialTime > 0) {
            console.log("Restoring time:", initialTime);
            if (audioPlayer.readyState >= 1) audioPlayer.currentTime = initialTime;
            else audioPlayer.addEventListener('loadedmetadata', () => { audioPlayer.currentTime = initialTime; }, { once: true });
        }

        function updateHighlight() {
            const currentTime = audioPlayer.currentTime;

            // Throttle save
            const now = Date.now();
            if (now - lastSaveTime > 1000) {
                localStorage.setItem(STORAGE_KEY, currentTime);
                lastSaveTime = now;
            }

            // 1. Determine correct chunk
            let pIdx = findClosestParagraphIndex(currentTime);
            // Check if pIdx is valid for this time (or close enough)
            // If currentTime is way past the end of pIdx (gap), we might want next one?
            // "ClosestParagraphIndex" finds the one that started before `time`.
            // So if time is 100, and P1 is 0-50, P2 is 102-150.
            // pIdx(100) -> P1. But we are closer to P2?
            // Effectively, we just want to ensure we are showing the text that *covers* this time.

            let targetChunk = Math.floor(pIdx / CHUNK_SIZE);
            if (targetChunk !== currentChunkIndex) {
                renderChunk(targetChunk);
            }

            // 2. Highlight word in current chunk
            // We just query selector in the *current reduced DOM*, so it's fast.
            const wordSpans = document.getElementsByClassName('word');
            // optimization: binary search is still useful since we have data-start attributes.
            // But strict linear scan of 2000-3000 words (50 paragraphs) is trivial for JS.

            // Linear scan on reduced DOM is fine
            let activeSpan = null;
            // Optimization: Start from middle? No, standard loop.
            // Or use the binary search logic on 'wordsData' to find the index, then map to DOM?
            // "Find index in wordsData" -> check if that index is in rendered DOM range? 
            // Too complex mapping.
            // Simple: Just iterate spans?
            // With 50 paragraphs, maybe ~5000 words. 60Hz loop might be heavy?
            // Let's optimize: only re-scan if time changed significantly?
            // Or binary search the DOM elements based on data-start?

            // Let's use the binary search on DOM nodes method.
            let low = 0, high = wordSpans.length - 1;
            let found = -1;

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
                // Clear existing
                const prev = document.querySelector('.active-word');
                if (prev) prev.classList.remove('active-word');

                activeSpan = wordSpans[found];
                activeSpan.classList.add('active-word');

                // Visible check
                const rect = activeSpan.getBoundingClientRect();
                if (rect.top < 0 || rect.bottom > window.innerHeight) {
                    activeSpan.scrollIntoView({ behavior: 'auto', block: 'center' });
                }
            }
        }

        audioPlayer.addEventListener('timeupdate', updateHighlight);
        audioPlayer.addEventListener('seeked', updateHighlight);
        audioPlayer.addEventListener('pause', () => localStorage.setItem(STORAGE_KEY, audioPlayer.currentTime));
    }
});
