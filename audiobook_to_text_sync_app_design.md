# Audiobook to Text Sync App Design

## 1. Introduction

This document outlines the design for an pip install -r c:\Code\Lumi\sync_service\requirements.txt

## 2. Features

*   **Audio Playback**: Standard audio player controls (play, pause, seek, volume).
*   **Text Display**: Display the text of the book in a readable format.
*   **Real-time Synchronization**: Highlight words, sentences, or paragraphs in the text as they are being narrated in the audiobook.
*   **Cross-Platform Support**: The app will be available on iOS, Android, and the web.
*   **User Accounts**: Users can create accounts to save their progress and library.
*   **Library Management**: Users can add, remove, and organize their audiobooks.
*   **Offline Access**: Users can download audiobooks and text for offline use.
*   **Multi-format Support**: The app will support common audiobook (MP3, M4A, M4B) and e-book (EPUB, PDF) formats.

## 3. Architecture

The application will follow a client-server architecture.

*   **Frontend (Client)**: A mobile application (iOS/Android) and a web application.
*   **Backend (Server)**: A set of services to manage users, books, and synchronization data.
*   **Database**: To store user data, book metadata, and progress.
*   **Storage**: To store audiobook and e-book files.
*   **Synchronization Service**: A dedicated service to process and align audio with text.

```
+----------------+      +-----------------+      +----------------+
| Mobile App     |      | Web App         |      | Admin Dashboard|
+----------------+      +-----------------+      +----------------+
        |                      |                      |
        +----------------------+----------------------+
                               |
                       +-------v-------+
                       | API Gateway   |
                       +---------------+
                               |
        +----------------------+----------------------+
        |                      |                      |
+-------v--------+    +--------v-------+    +---------v------+
| User Service   |    | Book Service   |    | Sync Service   |
+----------------+    +----------------+    +----------------+
        |                      |                      |
+-------v--------+    +--------v-------+    +---------v------+
| Database       |    | Object Storage |    | Caching        |
| (PostgreSQL)   |    | (S3, GCS)      |    | (Redis)        |
+----------------+    +----------------+    +----------------+
```

## 4. Data Models

### User
-   `user_id` (PK)
-   `username`
-   `email`
-   `password_hash`
-   `created_at`

### Book
-   `book_id` (PK)
-   `title`
-   `author`
-   `cover_image_url`
-   `audio_file_url`
-   `text_file_url`
-   `sync_data_url` (JSON file with timestamps)
-   `uploaded_by` (FK to User)
-   `created_at`

### UserBookProgress
-   `progress_id` (PK)
-   `user_id` (FK to User)
-   `book_id` (FK to Book)
-   `current_position_seconds`
-   `last_updated`

## 5. API Design

### Authentication
-   `POST /api/auth/register`
-   `POST /api/auth/login`
-   `POST /api/auth/logout`

### Books
-   `GET /api/books`: Get a list of all books.
-   `GET /api/books/{book_id}`: Get details for a specific book.
-   `POST /api/books`: Upload a new book (audio + text).
-   `DELETE /api/books/{book_id}`: Delete a book.

### Progress
-   `GET /api/progress/{book_id}`: Get user's progress for a book.
-   `POST /api/progress/{book_id}`: Update user's progress.

## 6. Synchronization Implementation Details

The synchronization process is the core feature of the application, and its accuracy is critical for a good user experience. The `Sync Service` will use **`faster-whisper`**, a high-performance implementation of OpenAI's Whisper model, to generate highly precise word-level timestamps. The entire process is automated and broken down into the following pipeline:

1.  **File Upload and Pre-processing**:
    *   The user uploads an audiobook file (e.g., `book.mp3`) and a text file (e.g., `book.txt`).
    *   The `Sync Service` expects a clean, plain text format. If the original text is in another format (e.g., EPUB), it must be converted first.
    *   The plain text is then segmented into a structured list of paragraphs (based on double newlines), each with a unique ID. This is the **ground-truth text**.
    *   Example of segmented ground-truth text (`segments.json`):
        ```json
        [
          { "id": "p1", "text": "The quick brown fox jumps over the lazy dog." },
          { "id": "p2", "text": "This is the second sentence." }
        ]
        ```

2.  **Transcription with Word-Level Timestamps**:
    *   The audiobook file is loaded and processed by a `faster-whisper` model (`WhisperModel`).
    *   The model's `transcribe` method is called with `word_timestamps=True`. This generates a highly accurate transcription of the audio with precise `start` and `end` timestamps for every word.
    *   Example of `faster-whisper` word-level output (`transcribed_words.json`):
        ```json
        [
            {"word": " The", "start": 0.12, "end": 0.34, "probability": 0.99},
            {"word": " quick", "start": 0.35, "end": 0.67, "probability": 0.98},
            {"word": " brown", "start": 0.68, "end": 0.95, "probability": 0.98},
            // ... and so on for every transcribed word
        ]
        ```

3.  **Alignment and Sync Map Generation**:
    *   To efficiently map the ground-truth text to the transcription, the alignment is done segment by segment.
    *   For each ground-truth segment (e.g., a paragraph), a **search window** is defined in the transcribed word list. This window is typically larger than the segment itself (e.g., 3 times the number of words) to account for discrepancies.
    *   A text alignment algorithm (**Needleman-Wunsch**) is then used to find the best match between the ground-truth segment's words and the words within the search window of the transcription.
    *   Once the alignment is found, the `start` timestamp of the first matched word and the `end` timestamp of the last matched word in the window are extracted.
    *   This process is repeated for all segments, and a cursor is maintained to keep track of the position in the transcribed word list, ensuring each segment is processed sequentially.
    *   The final output is the `sync_map.json` file, which maps each segment ID to its corresponding start and end times.
    *   Example of the final `sync_map.json`:
        ```json
        {
          "p1": { "start": 0.12, "end": 2.51 },
          "p2": { "start": 2.63, "end": 4.25 }
        }
        ```

4.  **Client-side Implementation**:
    *   When a user plays an audiobook, the client application fetches the audio file, the segmented ground-truth text (`segments.json`), and the final synchronization map (`sync_map.json`).
    *   As the audio plays, the application monitors the `currentTime` of the audio player.
    *   It uses this `currentTime` to look up which segment should be active by finding the segment in `sync_map.json` where `currentTime` is between `start` and `end`.
    *   The application then applies a highlight to the corresponding text segment on the screen.

This detailed, multi-step process leverages the power of `faster-whisper` for accurate timestamp generation and a windowed Needleman-Wunsch algorithm for efficient and accurate alignment.

## 7. UI/UX

*   **Library View**: A grid or list of book covers that the user has added.
*   **Player View**:
    *   Audio player controls at the bottom.
    *   The book text displayed in a scrollable view.
    *   The currently playing sentence/paragraph will be highlighted.
    *   Tapping on a paragraph could seek the audio to that point.

## 8. Technology Stack

*   **Frontend**:
    *   **Mobile**: React Native or Flutter for cross-platform development.
    *   **Web**: React or Vue.js.
*   **Backend**:
    *   **Framework**: Node.js (Express), Python (Django/Flask), or Go.
    *   **Database**: PostgreSQL or MongoDB.
    *   **Cache**: Redis for caching session data and frequently accessed content.
*   **Storage**: AWS S3 or Google Cloud Storage for storing media files.
*   **Synchronization Service**:
    *   **Transcription and Alignment Engine**: The application will use **`faster-whisper`**. It is a reimplementation of OpenAI's Whisper model that is up to 4 times faster and uses less memory, while providing highly accurate word-level timestamps. It will be self-hosted and integrated directly into the `Sync Service`.
*   **DevOps**: Docker, Kubernetes, GitHub Actions for CI/CD.

## 9. Challenges

*   **Synchronization Accuracy**: The accuracy of the forced alignment engine is crucial. Inaccurate timestamps will lead to a poor user experience. Manual correction tools might be needed for publishers, especially for audio with background noise or unusual narration.
*   **Format Compatibility**: Handling various e-book and audiobook formats can be complex. A robust processing pipeline will be required to convert them to a standard format.
*   **Scalability**: The application needs to be able to handle a large number of users and books. This requires a scalable architecture for both the backend services and the media storage.
*   **Copyright and DRM**: The app must respect digital rights management (DRM) and copyright laws. This may involve integrating with third-party DRM solutions.
