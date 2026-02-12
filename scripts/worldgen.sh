
#!/bin/bash

#
# tModLoader World Generator
# 
# This script automates the process of generating multiple Terraria worlds
# using tModLoader. It handles downloading tModLoader, generating random
# world seeds, and creating worlds with automatic server shutdown.
#
# Usage: ./worldgen.sh [NUM_WORLDS]
#
# Arguments:
#   NUM_WORLDS  - Number of worlds to generate (default: 20)
#
# Examples:
#   ./worldgen.sh           # Generate 20 worlds (default)
#   ./worldgen.sh 50        # Generate 50 worlds
#
# Features:
#   - Automatic tModLoader download and installation
#   - Skip download if tModLoader is already installed
#   - Generate multiple worlds in one run
#   - Random world names and seeds
#   - Automatic server shutdown after each world generation
#   - Progress tracking and colored output
#   - Error handling and validation
#   - Non-interactive (suitable for automation)
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

TMODLOADER_VERSION="v2025.12.3.0"
TMODLOADER_URL="https://github.com/tModLoader/tModLoader/releases/download/${TMODLOADER_VERSION}/tModLoader.zip"
DOWNLOADS_DIR="./downloads"
WORLDGEN_DIR="./worldgen"
TMODLOADER_ZIP="${DOWNLOADS_DIR}/tModLoader.zip"
TMODLOADER_SERVER="${DOWNLOADS_DIR}/start-tModLoaderServer.sh"
LOGS_DIR="./logs"

# Default tModLoader world save location
DEFAULT_WORLD_DIR="$HOME/.local/share/Terraria/tModLoader/Worlds"

# World generation timeout (seconds)
GENERATION_TIMEOUT=600

# Default number of worlds to generate
DEFAULT_NUM_WORLDS=20

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

# Log an info message
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Log an error message
log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Log a warning message
log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Log a debug message
log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check if tModLoader is already installed
check_tmodloader_installation() {
    if [[ -f "${TMODLOADER_SERVER}" ]]; then
        log_info "tModLoader installation found at: ${TMODLOADER_SERVER}"
        return 0
    else
        return 1
    fi
}

# Download and extract tModLoader
download_tmodloader() {
    log_info "Downloading tModLoader ${TMODLOADER_VERSION}..."
    
    # Create downloads directory if it doesn't exist
    mkdir -p "${DOWNLOADS_DIR}"
    
    # Download tModLoader
    if ! curl -Lo "${TMODLOADER_ZIP}" "${TMODLOADER_URL}"; then
        log_error "Failed to download tModLoader from ${TMODLOADER_URL}"
        exit 1
    fi
    
    log_info "Extracting tModLoader..."
    cd "${DOWNLOADS_DIR}"
    
    # Extract the zip file
    if ! unzip -o ./tModLoader.zip; then
        log_error "Failed to extract tModLoader"
        cd ..
        exit 1
    fi
    
    # Clean up zip file
    rm ./tModLoader.zip
    
    # Make server script executable
    chmod +x ./start-tModLoaderServer.sh
    
    cd ..
    log_info "tModLoader installed successfully"
}

# Generate a random world name
generate_world_name() {
    # Generate a random alphanumeric string (16 characters)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local random_suffix=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
    echo "World_${timestamp}_${random_suffix}"
}

# Generate a random seed
generate_seed() {
    # Generate a random number seed (using multiple RANDOM calls for better randomness)
    echo $((RANDOM * RANDOM + RANDOM))
}

# Generate a single world
generate_world() {
    local world_name=$1
    local seed=$2
    local world_num=$3
    local total_worlds=$4
    
    echo "----------------------------------------"
    log_info "[$world_num/$total_worlds] Generating world: ${world_name}"
    log_debug "Seed: ${seed}"
    
    # Create directories if they don't exist
    mkdir -p "${WORLDGEN_DIR}"
    mkdir -p "${LOGS_DIR}"
    
    # Get absolute path for world file
    local worldgen_abs_path=$(cd "${WORLDGEN_DIR}" && pwd)
    local world_file_path="${worldgen_abs_path}/${world_name}.wld"
    
    # Set up log file for this world
    local log_file="${LOGS_DIR}/${world_name}.log"
    
    # Start the server in the background and capture its PID
    cd "${DOWNLOADS_DIR}"
    log_debug "Starting server with command: ./start-tModLoaderServer.sh -nosteam -autocreate 3 -seed ${seed} -worldname ${world_name} -world ${world_file_path}"
    ./start-tModLoaderServer.sh -nosteam -autocreate 3 -seed "${seed}" -worldname "${world_name}" -world "${world_file_path}" > "../${log_file}" 2>&1 &
    local server_pid=$!
    cd ..
    
    log_debug "Server started with PID: ${server_pid}"
    log_debug "Server output: ${log_file}"
    log_debug "Expected world path: ${world_file_path}"
    
    # Wait a moment for the server to initialize
    sleep 5
    
    # Check if the process is still running
    if ! kill -0 $server_pid 2>/dev/null; then
        log_error "Server process died immediately after starting"
        log_error "Check log file for details: ${log_file}"
        if [[ -f "${log_file}" ]]; then
            log_error "Last 20 lines of log:"
            tail -n 20 "${log_file}" | while IFS= read -r line; do
                echo "  | $line"
            done
        fi
        return 1
    fi
    
    # Wait for world generation to complete (check for world file existence)
    local waited=0
    local world_found=false
    while [[ $waited -lt $GENERATION_TIMEOUT ]]; do
        # Check both the specified location and default tModLoader location
        if [[ -f "${world_file_path}" ]]; then
            world_found=true
            log_debug "World file found at specified location: ${world_file_path}"
            break
        elif [[ -f "${DEFAULT_WORLD_DIR}/${world_name}.wld" ]]; then
            world_found=true
            log_debug "World file found at default location: ${DEFAULT_WORLD_DIR}/${world_name}.wld"
            log_debug "Moving to target location: ${world_file_path}"
            # Move the world files (there are usually 2-3 associated files)
            mv "${DEFAULT_WORLD_DIR}/${world_name}".* "${worldgen_abs_path}/" 2>/dev/null || true
            break
        fi
        
        sleep 5
        waited=$((waited + 5))
        
        # Check if the process is still running
        if ! kill -0 $server_pid 2>/dev/null; then
            log_error "Server process terminated unexpectedly after ${waited}s"
            log_error "Check log file for details: ${log_file}"
            if [[ -f "${log_file}" ]]; then
                log_error "Last 30 lines of log:"
                tail -n 30 "${log_file}" | while IFS= read -r line; do
                    echo "  | $line"
                done
            fi
            return 1
        fi
        
        # Show progress every 30 seconds
        if [[ $((waited % 30)) -eq 0 ]] && [[ $waited -gt 0 ]]; then
            log_debug "Still generating... (${waited}s elapsed, process still running)"
        fi
    done
    
    # Check if world was generated successfully
    if [[ "$world_found" == true ]] && [[ -f "${world_file_path}" ]]; then
        log_info "World file created successfully"
        
        # Kill the server process gracefully
        log_debug "Shutting down server..."
        kill $server_pid 2>/dev/null || true
        
        # Wait a moment for graceful shutdown
        sleep 2
        
        # Force kill if still running
        kill -9 $server_pid 2>/dev/null || true
        wait $server_pid 2>/dev/null || true
        
        log_info "[$world_num/$total_worlds] World '${world_name}' generated successfully ✓"
        return 0
    else
        log_error "World generation timed out after ${GENERATION_TIMEOUT} seconds"
        log_error "Check log file for details: ${log_file}"
        
        # Show tail of log file for debugging
        if [[ -f "${log_file}" ]]; then
            log_error "Last 30 lines of log:"
            tail -n 30 "${log_file}" | while IFS= read -r line; do
                echo "  | $line"
            done
        fi
        
        # Kill the server process
        kill -9 $server_pid 2>/dev/null || true
        wait $server_pid 2>/dev/null || true
        
        return 1
    fi
}

# =============================================================================
# Main Function
# =============================================================================

main() {
    # Parse command-line arguments
    local num_worlds="${1:-$DEFAULT_NUM_WORLDS}"
    
    # Validate input
    if ! [[ "$num_worlds" =~ ^[0-9]+$ ]] || [[ "$num_worlds" -lt 1 ]]; then
        log_error "Invalid argument. NUM_WORLDS must be a positive number."
        log_error "Usage: $0 [NUM_WORLDS]"
        exit 1
    fi
    
    log_info "World generation started: ${num_worlds} world(s)"
    
    # Check if tModLoader is installed, download if not
    if ! check_tmodloader_installation; then
        log_warn "tModLoader not found - initiating download"
        download_tmodloader
    else
        log_info "Using existing tModLoader installation (skipping download)"
    fi
    
    log_info "Preparing to generate ${num_worlds} world(s)..."
    echo
    
    # Track statistics
    local success_count=0
    local failure_count=0
    
    # Generate worlds
    for ((i=1; i<=num_worlds; i++)); do
        world_name=$(generate_world_name)
        seed=$(generate_seed)
        
        if generate_world "${world_name}" "${seed}" "${i}" "${num_worlds}"; then
            success_count=$((success_count + 1))
        else
            failure_count=$((failure_count + 1))
            log_error "Failed to generate world ${i}/${num_worlds}"
        fi
        
        echo
        
        # Add a small delay between world generations
        if [[ $i -lt $num_worlds ]]; then
            sleep 2
        fi
    done
    
    # Summary
    echo "========================================"
    echo "  Generation Summary"
    echo "========================================"
    log_info "Total worlds requested: ${num_worlds}"
    log_info "Successfully generated: ${success_count}"
    if [[ $failure_count -gt 0 ]]; then
        log_warn "Failed: ${failure_count}"
        log_warn "Check log files in ${LOGS_DIR}/ for details"
    fi
    log_info "Output directory: ${WORLDGEN_DIR}/"
    log_info "Log directory: ${LOGS_DIR}/"
    echo
    
    # List generated worlds
    if [[ $success_count -gt 0 ]]; then
        log_info "Generated world files:"
        ls -lh "${WORLDGEN_DIR}"/*.wld 2>/dev/null || true
    fi
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Run main function with all command-line arguments
main "$@"
