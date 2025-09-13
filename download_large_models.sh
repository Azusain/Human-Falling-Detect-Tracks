#!/bin/bash
# Download large model files from Google Drive during Docker build

set -e

echo "Downloading large model files from Google Drive..."

# Function to download from Google Drive using gdown
download_from_gdrive() {
    local file_id="$1"
    local output_path="$2"
    local file_name=$(basename "$output_path")
    
    echo "Downloading $file_name..."
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")"
    
    # Download using gdown
    if command -v gdown &> /dev/null; then
        gdown --id "$file_id" -O "$output_path" --no-check-certificate
    else
        echo "Error: gdown is not installed. Please install it first: pip install gdown"
        exit 1
    fi
    
    if [[ -f "$output_path" ]]; then
        file_size=$(du -h "$output_path" | cut -f1)
        echo "✓ Downloaded $file_name ($file_size)"
    else
        echo "✗ Failed to download $file_name"
        exit 1
    fi
}

# Download large model files that exceed GitHub's 100MB limit
echo "Downloading SPPE models..."

# fast_res50_256x192.pth (155.07 MB)
download_from_gdrive "1e85yn3Mj6U_fbZBcqZ2BzVdVbjB4Gvyo" "Models/sppe/fast_res50_256x192.pth"

# fast_res101_320x256.pth (227.81 MB) 
download_from_gdrive "1HuOEuAVVqK47Lmi35AlivMW3eQgUfOAL" "Models/sppe/fast_res101_320x256.pth"

echo "All large model files downloaded successfully!"
