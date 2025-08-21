#!/bin/bash
# Build script for frontend on Render

cd frontend
npm install
npm run build
cd ..

# Copy build to root for easier access
cp -r frontend/build ./build
