FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 1. Copies your app.py, templates, and college_info.txt inside
COPY . .

# 2. ⚡ THE AUTOMATED FIX: Build the database right inside the cloud container
RUN python create_db.py

EXPOSE 8080

CMD ["python", "app.py"]
