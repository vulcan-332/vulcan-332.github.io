#!/bin/bash
# Script to run the Jekyll site locally using the configured Conda environment.

# Ensure the script stops on error
set -e

echo "Starting Jekyll site..."
echo "Open http://127.0.0.1:4000 in your browser once the server starts."

# Run jekyll serve within the conda environment using absolute path to bundle
conda run -n jekyll-env /Users/kmirza/miniconda3/envs/jekyll-env/bin/bundle exec jekyll serve
