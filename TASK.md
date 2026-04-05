# TTS Service Migration - Summary

## Current State
- UI server is in Python
- Server "tardis" (see TARDIS.md) has pm2 for process management

## Goal
1. Translate Python UI server to Node.js
2. Kill old Python server on tardis
3. Start new Node.js service with pm2

## Requirements
- Node.js wrapper for Kokoro-82M TTS
- Process management via pm2
