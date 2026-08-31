# ============================================================
# JobHunter - Phase 1 Environment Layer
# Debian bookworm-slim | Python 3.11 | Node.js LTS | LaTeX
# ============================================================
FROM python:3.11-slim-bookworm

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NODE_MAJOR=20 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# ------------------------------------------------------------
# 1. Core OS tooling + curl/gnupg for the NodeSource repo
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2. Node.js LTS (via NodeSource)
# ------------------------------------------------------------
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version && npm --version

# ------------------------------------------------------------
# 3. Minimal viable TeX stack for ATS-friendly PDF generation
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        lmodern \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 4. Playwright / Chromium system dependencies (headless)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxcb1 \
        libxkbcommon0 \
        libx11-6 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 5. Python dependencies
# ------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install Playwright's Chromium browser binary
RUN python -m playwright install chromium

# ------------------------------------------------------------
# 6. The "Career-Ops" Dynamic Pull
#    Clones Santiago's latest engine into /app/career-ops
# ------------------------------------------------------------
WORKDIR /app
RUN npx --yes @santifer/career-ops init

# ------------------------------------------------------------
# 7. Application source
# ------------------------------------------------------------
WORKDIR /app
COPY . .

EXPOSE 3000

# FastAPI orchestrator entrypoint (built in a later phase)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3000"]
