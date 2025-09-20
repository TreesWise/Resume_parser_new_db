#!/bin/bash
set -e  # Exit immediately if a command fails

echo "Updating packages..."
apt-get update -y

echo "Installing dependencies..."
apt-get install -y --no-install-recommends \
    libreoffice \
    curl \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    unixodbc-dev \
    build-essential

echo "Adding Microsoft package repository..."
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
# Detect Ubuntu/Debian version and add the repo
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$VERSION_ID" == "22.04" ]]; then
        curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
    else
        curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
    fi
fi

echo "Installing Microsoft ODBC Driver 18..."
apt-get update -y
ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc

echo "Verifying pyodbc..."
python -c "import pyodbc; print('Available ODBC drivers:', pyodbc.drivers())" || pip install --no-cache-dir pyodbc

echo "Starting FastAPI app with Gunicorn..."
gunicorn -w 2 -b 0.0.0.0:8000 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 240 \
    main:app

