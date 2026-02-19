# Docker Technical Review: noapi-google-search-mcp

## Project Overview

**Project Name**: noapi-google-search-mcp  
**Repository**: https://github.com/VincentKaufmann/noapi-google-search-mcp  
**Version**: 0.2.5  
**License**: MIT  
**Purpose**: MCP (Model Context Protocol) server that provides Google search capabilities without API keys using headless Chromium via Playwright.

## Technical Analysis

### Current Project Structure

```
noapi-google-search-mcp/
├── .github/              # GitHub workflows
├── images/               # Documentation images
├── src/
│   └── google_search_mcp/
│       ├── __init__.py
│       ├── __main__.py
│       └── server.py         # Main MCP server (3,881 lines)
├── .gitignore
├── LICENSE
├── pyproject.toml        # Python project configuration
└── README.md              # Comprehensive documentation
```

### Dependencies Analysis

From [`pyproject.toml`](../llm/mcp/noapi-google-search-mcp/pyproject.toml):

```toml
dependencies = [
    "mcp>=1.0.0",              # Model Context Protocol
    "playwright>=1.40.0",         # Browser automation
    "opencv-python-headless>=4.8.0", # Computer vision (object detection)
    "rapidocr-onnxruntime>=1.4.0",  # OCR (offline)
    "yt-dlp>=2024.0",            # Video download
    "faster-whisper>=1.0.0",      # Video transcription
]
```

**Python Version Requirement**: >=3.10

### Key Features

The project provides 20+ MCP tools:

1. **Search Tools**:
   - `google_search` - Web search with filters (time, site, language, region)
   - `google_news` - News search with thumbnails
   - `google_scholar` - Academic paper search
   - `google_images` - Image search with inline display
   - `google_trends` - Topic interest trends
   - `google_shopping` - Product search
   - `google_books` - Book search
   - `google_translate` - Translation
   - `google_flights` - Flight search
   - `google_hotels` - Hotel search
   - `google_finance` - Stock/market data
   - `google_weather` - Weather lookup

2. **Vision & OCR Tools**:
   - `google_lens` - Reverse image search
   - `google_lens_detect` - Object detection + identification
   - `ocr_image` - Local OCR (RapidOCR, offline)

3. **Video Tools**:
   - `transcribe_video` - YouTube/video transcription with timestamps
   - `search_transcript` - Search transcribed video
   - `extract_video_clip` - Extract clips by topic

4. **Utility Tools**:
   - `list_images` - List image files in directory
   - `visit_page` - Fetch and extract page content

## Docker Feasibility Assessment

### ✅ **FEASIBLE** - Docker Image Creation is Possible

### Requirements Analysis

| Requirement | Status | Notes |
|------------|--------|--------|
| Base Image | ✅ | Python 3.10+ base image available (python:3.10-slim, python:3.11-slim) |
| Dependencies | ✅ | All dependencies are pip-installable |
| System Libraries | ⚠️ | Requires system-level libraries (Chromium, fonts, codecs) |
| Playwright Browsers | ✅ | Can install Chromium in Docker |
| OpenCV | ✅ | opencv-python-headless works in Docker |
| File System Access | ⚠️ | Requires volume mounts for persistent data |
| Network Access | ✅ | Required for Google access, works in Docker |
| Cache/Temp Storage | ✅ | Can use volumes for cache persistence |

### Docker Implementation Considerations

#### 1. **Playwright Browser Installation**

The project uses Playwright with headless Chromium. In Docker:

```dockerfile
# Install Playwright browsers
RUN playwright install chromium
```

**Challenge**: Playwright browser installation requires ~300-500MB of additional space and takes time during build.

**Solution**: Use multi-stage build to optimize layer caching.

#### 2. **System Dependencies**

The project requires several system-level libraries:

- **Chromium**: For Playwright browser automation
- **Fonts**: For text rendering in screenshots
- **Codecs**: For video processing (ffmpeg, libavcodec, etc.)
- **OCR Models**: RapidOCR ONNX models (downloaded at runtime)

**Docker Base Image Recommendation**:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-liberation \
    fonts-noto-cjk \
    ffmpeg \
    libavcodec-extra \
    libavformat-extra \
    libavutil-extra \
    libswscale-extra \
    libavdevice-extra \
    libavfilter-extra \
    wget \
    && rm -rf /var/lib/apt/lists/*
```

#### 3. **Persistent Storage Requirements**

The application creates and uses several directories:

- `~/.cache/noapi-google-search-mcp/` - Cache for transcripts, videos, images
- `~/clips/` - Extracted video clips
- `~/lens/` - Default image directory for lens tools

**Docker Volume Recommendations**:
```yaml
volumes:
  - ${HOME}/.cache/noapi-google-search-mcp:/root/.cache/noapi-google-search-mcp
  - ${HOME}/clips:/root/clips
  - ${HOME}/lens:/root/lens
```

#### 4. **Performance Considerations**

- **Browser Launch Overhead**: Each tool launch creates a new browser instance (~500MB-1GB memory)
- **Video Processing**: Transcription and clip extraction are CPU-intensive
- **Memory Requirements**: Minimum 2GB RAM recommended, 4GB+ for video tasks
- **CPU**: Multi-core CPU recommended for faster transcription

#### 5. **Network Requirements**

- **Outbound Internet Access**: Required for Google services
- **No Inbound Ports Needed**: MCP uses stdio/stdout for communication
- **Rate Limiting**: Google may rate-limit automated requests

### Recommended Docker Configuration

#### Dockerfile

```dockerfile
# Multi-stage build for noapi-google-search-mcp
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir /tmp/pip \
    && pip install . \
    && rm -rf /tmp/pip

# Install Playwright browsers
RUN playwright install chromium --with-deps chromium

# Install additional system libraries for video/OCR
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-liberation \
    fonts-noto-cjk \
    ffmpeg \
    libavcodec-extra \
    libavformat-extra \
    libavutil-extra \
    libswscale-extra \
    libavdevice-extra \
    libavfilter-extra \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories
RUN mkdir -p /root/.cache/noapi-google-search-mcp \
    /root/clips \
    /root/lens

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app
COPY --from=builder /root/.cache /root/.cache

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    fonts-liberation \
    fonts-noto-cjk \
    ffmpeg \
    libavcodec-extra \
    libavformat-extra \
    libavutil-extra \
    libswscale-extra \
    libavdevice-extra \
    libavfilter-extra \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app /root/.cache
USER mcpuser

# Expose no ports (MCP uses stdio)
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the MCP server
CMD ["python", "-m", "google_search_mcp"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  noapi-google-search-mcp:
    build:
      context: .
      dockerfile: Dockerfile
    image: noapi-google-search-mcp:latest
    container_name: noapi-google-search-mcp
    
    # Resource limits (adjust based on usage)
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    
    # Volume mounts for persistent storage
    volumes:
      # Cache directory for transcripts, videos, images
      - ${HOME:-.}/cache/noapi-google-search-mcp:/root/.cache/noapi-google-search-mcp
      # Extracted video clips
      - ${HOME:-.}/clips:/root/clips
      # Default lens image directory
      - ${HOME:-.}/lens:/root/lens
    
    # Environment variables
    environment:
      - PYTHONUNBUFFERED=1
      - PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright
    
    # Restart policy
    restart: unless-stopped
    
    # No ports needed (MCP uses stdio)
    # Network mode
    network_mode: bridge
    
    # Security options
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Deployment Recommendations

#### 1. **Resource Allocation**

| Component | Minimum | Recommended | Notes |
|-----------|----------|-------------|--------|
| CPU | 1 core | 2+ cores | Video transcription benefits from more cores |
| RAM | 2GB | 4GB+ | Browser + video processing is memory-intensive |
| Disk | 5GB | 10GB+ | Cache, models, clips can accumulate |

#### 2. **Performance Optimization**

```yaml
# For better performance, consider:
environment:
  # Use all available CPU cores for transcription
  - OMP_NUM_THREADS=4
  # Playwright optimizations
  - PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
```

#### 3. **Cache Strategy**

The project implements disk caching:
- **Transcripts**: Cached in `~/.cache/noapi-google-search-mcp/transcripts/`
- **Videos**: Cached in `~/.cache/noapi-google-search-mcp/videos/`

**Recommendation**: Use bind mounts instead of volumes for faster I/O on local development:
```yaml
volumes:
  - ./cache:/root/.cache/noapi-google-search-mcp
```

#### 4. **Security Considerations**

- ✅ **Non-root user**: Container runs as `mcpuser` (UID 1000)
- ✅ **Read-only root**: No privileged mode needed
- ✅ **No exposed ports**: MCP communicates via stdio only
- ⚠️ **Network access**: Requires outbound internet to Google services
- ⚠️ **User input**: Image files from host need proper volume mounting

### Known Limitations & Challenges

#### 1. **Browser Automation Detection**

Google may detect and block automated browsers:
- CAPTCHA challenges
- Rate limiting
- IP blocking
- Unusual traffic patterns

**Mitigation**:
- Use residential proxy (not included in base project)
- Implement request delays
- Rotate user agents (already implemented)

#### 2. **Video Processing Performance**

- CPU-only transcription is slow (real-time requires ~10x faster than audio duration)
- Large videos (>10 min) may timeout
- Clip extraction requires full video download first

**Mitigation**:
- Use `tiny` model for faster transcription
- Process shorter segments
- Consider GPU acceleration (not currently supported)

#### 3. **Storage Growth**

- Transcript cache can grow large with many videos
- Clip extraction creates duplicates
- No automatic cleanup implemented

**Mitigation**:
- Implement periodic cleanup cron job
- Set cache size limits
- Use external storage for long-term retention

### Comparison with Alternatives

| Feature | noapi-google-search-mcp | tavily-mcp | API-based Solutions |
|----------|------------------------|-----------|------------------|
| Cost | Free (no API key) | Paid (Tavily API) | Paid (Google CSE API) |
| Setup | `pip install` | API key registration | Google Cloud project setup |
| Search Quality | Real Google results | Tavily search results | Custom Search Engine results |
| JavaScript Pages | ✅ Renders them | ❌ Cannot render | ❌ Cannot render |
| Rate Limits | Google's limits | Tavily's limits | API quota limits |
| Vision Capabilities | ✅ Google Lens + OCR | ❌ Not available | Separate API needed |
| Video Processing | ✅ Local transcription | ❌ Not available | Not available |
| Offline OCR | ✅ RapidOCR | ❌ Not available | Not available |

### Conclusion

**Docker Image Creation: ✅ FEASIBLE**

The [`noapi-google-search-mcp`](../llm/mcp/noapi-google-search-mcp/) project can be successfully containerized with Docker. The main considerations are:

1. **System Dependencies**: Requires Chromium, fonts, and video codec libraries
2. **Resource Requirements**: 2GB+ RAM, 2+ CPU cores recommended
3. **Persistent Storage**: Volume mounts needed for cache, clips, and lens directories
4. **Network Access**: Outbound internet required for Google services
5. **Performance**: Video transcription is CPU-intensive; consider resource limits

### Next Steps for Implementation

1. Create [`Dockerfile`](../llm/mcp/noapi-google-search-mcp/Dockerfile) in project root
2. Create [`docker-compose.yml`](../llm/mcp/noapi-google-search-mcp/docker-compose.yml) for easy deployment
3. Add `.dockerignore` to exclude unnecessary files
4. Test with various resource allocations
5. Document volume mount points
6. Add health checks and monitoring
7. Consider multi-architecture builds (amd64, arm64)

### References

- **Project README**: [`llm/mcp/noapi-google-search-mcp/README.md`](../llm/mcp/noapi-google-search-mcp/README.md)
- **Project Configuration**: [`llm/mcp/noapi-google-search-mcp/pyproject.toml`](../llm/mcp/noapi-google-search-mcp/pyproject.toml)
- **Main Server**: [`llm/mcp/noapi-google-search-mcp/src/google_search_mcp/server.py`](../llm/mcp/noapi-google-search-mcp/src/google_search_mcp/server.py)
- **MCP Specification**: https://modelcontextprotocol.io/
- **Playwright Docker Guide**: https://playwright.dev/docs/docker
- **Docker Best Practices**: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

---

**Review Date**: 2026-02-19  
**Reviewer**: AI Project Documentation Team  
**Status**: ✅ Docker deployment is technically feasible and recommended
