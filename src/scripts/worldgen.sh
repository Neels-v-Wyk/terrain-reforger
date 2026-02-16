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

# Default tModLoader world save location (platform-specific)
if [[ "$(uname)" == "Darwin" ]]; then
    DEFAULT_WORLD_DIR="$HOME/Library/Application Support/Terraria/tModLoader/Worlds"
else
    DEFAULT_WORLD_DIR="$HOME/.local/share/Terraria/tModLoader/Worlds"
fi

# World generation timeout (seconds)
GENERATION_TIMEOUT=600

# Default number of worlds to generate
DEFAULT_NUM_WORLDS=20

# Parallel generation settings
MAX_PARALLEL_JOBS=4

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
    # Generate a unique name using timestamp + nanoseconds + random suffix
    local timestamp=$(date +%Y%m%d_%H%M%S)
    # Use nanoseconds for sub-second uniqueness, plus process ID and random for extra entropy
    local nano=$(date +%N 2>/dev/null || echo "$$")
    local random_suffix=$(LC_ALL=C tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 8)
    echo "World_${timestamp}_${nano:0:4}_${random_suffix}"
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
    local port=${5:-7777}
    
    echo "----------------------------------------"
    log_info "[$world_num/$total_worlds] Generating world: ${world_name}"
    log_debug "Seed: ${seed}, Port: ${port}"
    
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
    log_debug "Starting server with command: ./start-tModLoaderServer.sh -nosteam -autocreate 3 -seed ${seed} -worldname ${world_name} -world ${world_file_path} -port ${port}"
    ./start-tModLoaderServer.sh -nosteam -autocreate 3 -seed "${seed}" -worldname "${world_name}" -world "${world_file_path}" -port "${port}" > "../${log_file}" 2>&1 &
    local server_pid=$!
    cd ..
    
    log_debug "Server started with PID: ${server_pid}"
    log_debug "Server output: ${log_file}"
    log_debug "Expected world path: ${world_file_path}"
    
    # Wait a moment for the server to initialize
    sleep 5
    
    # Check if the process is still running (early crash before world gen started)
    if ! kill -0 $server_pid 2>/dev/null; then
        # Process died early - check if it maybe already generated the world anyway
        sleep 2
        if [[ -f "${world_file_path}" ]] || [[ -f "${DEFAULT_WORLD_DIR}/${world_name}.wld" ]]; then
            log_debug "Server crashed but world file exists - continuing"
        else
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
    fi
    
    # Wait for world generation to complete by monitoring log for "Server started" or world file existence
    local waited=0
    local server_ready=false
    while [[ $waited -lt $GENERATION_TIMEOUT ]]; do
        # Check if log contains "Server started" which indicates world gen is complete
        if [[ -f "${log_file}" ]] && grep -q "Server started" "${log_file}" 2>/dev/null; then
            server_ready=true
            log_debug "Server started message detected in log - world generation complete"
            break
        fi
        
        # Also check if world file exists (server might have crashed after saving)
        if [[ -f "${world_file_path}" ]] || [[ -f "${DEFAULT_WORLD_DIR}/${world_name}.wld" ]]; then
            server_ready=true
            log_debug "World file detected - generation complete (server may have crashed after)"
            break
        fi
        
        sleep 5
        waited=$((waited + 5))
        
        # Check if the process is still running
        if ! kill -0 $server_pid 2>/dev/null; then
            # Server crashed - but check if world was generated before crash
            sleep 2
            if [[ -f "${world_file_path}" ]] || [[ -f "${DEFAULT_WORLD_DIR}/${world_name}.wld" ]]; then
                server_ready=true
                log_debug "Server crashed but world file exists - treating as success"
                break
            fi
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
    
    # Kill the server process immediately now that we know world gen is done
    if [[ "$server_ready" == true ]]; then
        log_debug "Shutting down server (PID: ${server_pid})..."
        
        # Kill the process tree (server may spawn child processes)
        pkill -P $server_pid 2>/dev/null || true
        kill $server_pid 2>/dev/null || true
        
        # Wait a moment for graceful shutdown
        sleep 1
        
        # Force kill if still running
        pkill -9 -P $server_pid 2>/dev/null || true
        kill -9 $server_pid 2>/dev/null || true
        wait $server_pid 2>/dev/null || true
        
        # Give filesystem time to sync
        sleep 1
        
        # Find and move the world file from default location to our worldgen directory
        local world_found=false
        if [[ -f "${world_file_path}" ]]; then
            world_found=true
            log_debug "World file found at specified location: ${world_file_path}"
        elif [[ -f "${DEFAULT_WORLD_DIR}/${world_name}.wld" ]]; then
            world_found=true
            log_debug "World file found at default location: ${DEFAULT_WORLD_DIR}/${world_name}.wld"
            log_debug "Moving to target location: ${world_file_path}"
            # Move the world files (there are usually 2-3 associated files)
            mv "${DEFAULT_WORLD_DIR}/${world_name}".* "${worldgen_abs_path}/" 2>/dev/null || true
        fi
        
        if [[ "$world_found" == true ]] && [[ -f "${world_file_path}" ]]; then
            log_info "[$world_num/$total_worlds] World '${world_name}' generated successfully ✓"
            return 0
        else
            log_error "World file not found after generation"
            log_error "Checked: ${world_file_path}"
            log_error "Checked: ${DEFAULT_WORLD_DIR}/${world_name}.wld"
            return 1
        fi
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
        pkill -P $server_pid 2>/dev/null || true
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
    local parallel="${2:-1}"  # Number of parallel jobs (default: 1 = sequential)
    
    # Validate input
    if ! [[ "$num_worlds" =~ ^[0-9]+$ ]] || [[ "$num_worlds" -lt 1 ]]; then
        log_error "Invalid argument. NUM_WORLDS must be a positive number."
        log_error "Usage: $0 [NUM_WORLDS] [PARALLEL_JOBS]"
        exit 1
    fi
    
    # Validate parallel jobs
    if ! [[ "$parallel" =~ ^[0-9]+$ ]] || [[ "$parallel" -lt 1 ]]; then
        parallel=1
    fi
    
    # Cap parallel jobs
    if [[ "$parallel" -gt "$MAX_PARALLEL_JOBS" ]]; then
        log_warn "Limiting parallel jobs to ${MAX_PARALLEL_JOBS} (requested: ${parallel})"
        parallel=$MAX_PARALLEL_JOBS
    fi
    
    log_info "World generation started: ${num_worlds} world(s), ${parallel} parallel job(s)"
    
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
    
    if [[ "$parallel" -eq 1 ]]; then
        # Sequential generation (original behavior)
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
    else
        # Parallel generation
        log_info "Running in parallel mode with ${parallel} concurrent jobs"
        
        # Create a temporary directory for job tracking
        local job_dir=$(mktemp -d)
        
        # Generate worlds in parallel batches
        local running_jobs=0
        local completed=0
        declare -a pids=()
        declare -a world_nums=()
        declare -a port_slots=()
        
        # Initialize port slots (track which ports are free)
        for ((p=0; p<parallel; p++)); do
            port_slots[$p]=0  # 0 means free
        done
        
        # Find a free port slot
        find_free_port_slot() {
            for ((p=0; p<parallel; p++)); do
                if [[ ${port_slots[$p]} -eq 0 ]]; then
                    echo $p
                    return
                fi
            done
            echo 0  # fallback
        }
        
        for ((i=1; i<=num_worlds; i++)); do
            world_name=$(generate_world_name)
            seed=$(generate_seed)
            
            # Find a free port slot
            local slot=$(find_free_port_slot)
            local port=$((7777 + slot))
            port_slots[$slot]=1  # mark as in use
            
            # Start world generation in background
            (
                if generate_world "${world_name}" "${seed}" "${i}" "${num_worlds}" "${port}"; then
                    touch "${job_dir}/success_${i}" 2>/dev/null || true
                else
                    touch "${job_dir}/failure_${i}" 2>/dev/null || true
                fi
            ) &
            
            local job_pid=$!
            pids+=($job_pid)
            world_nums+=($i)
            running_jobs=$((running_jobs + 1))
            
            # Wait if we've hit the parallel limit
            if [[ $running_jobs -ge $parallel ]]; then
                # Wait for any job to complete
                wait -n 2>/dev/null || wait ${pids[0]}
                running_jobs=$((running_jobs - 1))
                
                # Clean up completed PIDs and free their port slots
                local new_pids=()
                local idx=0
                for pid in "${pids[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        new_pids+=($pid)
                    else
                        # Free the port slot for completed job
                        local completed_slot=$((idx % parallel))
                        port_slots[$completed_slot]=0
                    fi
                    idx=$((idx + 1))
                done
                pids=()
                if [[ ${#new_pids[@]} -gt 0 ]]; then
                    pids=("${new_pids[@]}")
                fi
                
                # Reset all port slots if no jobs running (safety)
                if [[ ${#pids[@]} -eq 0 ]]; then
                    for ((p=0; p<parallel; p++)); do
                        port_slots[$p]=0
                    done
                fi
            fi
        done
        
        # Wait for remaining jobs
        log_info "Waiting for remaining jobs to complete..."
        if [[ ${#pids[@]} -gt 0 ]]; then
            for pid in "${pids[@]}"; do
                wait $pid 2>/dev/null || true
            done
        fi
        
        # Give a moment for all touch commands to complete
        sleep 2
        
        # Count results
        success_count=$(ls "${job_dir}"/success_* 2>/dev/null | wc -l | tr -d ' ')
        failure_count=$(ls "${job_dir}"/failure_* 2>/dev/null | wc -l | tr -d ' ')
        
        # Clean up temp directory
        rm -rf "${job_dir}" 2>/dev/null || true
    fi
    
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
