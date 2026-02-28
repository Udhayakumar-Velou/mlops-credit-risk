# Use official uv image (recommended)
FROM ghcr.io/astral-sh/uv:python3.9-bookworm

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]