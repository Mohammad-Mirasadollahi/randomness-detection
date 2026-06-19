#!/usr/bin/env bash
# Smoke test for install.sh — isolated temp install, real scoring samples, no mocks.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR=""
PORT=""
declare -A CLI_BACKUPS=()

log() { printf '[install-smoke] %s\n' "$*" >&2; }
fail() { printf '[install-smoke] FAIL: %s\n' "$*" >&2; exit 1; }
pass() { printf '[install-smoke] PASS: %s\n' "$*" >&2; }

cleanup() {
    if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
        if [[ -f "$WORK_DIR/.run/api.pid" ]]; then
            (cd "$WORK_DIR" && ./install.sh --stop >/dev/null 2>&1) || true
        fi
        rm -rf "$WORK_DIR"
    fi
    local name path
    for name in "${!CLI_BACKUPS[@]}"; do
        path="${CLI_BACKUPS[$name]:-}"
        if [[ -n "$path" && -f "$path" ]]; then
            cp -a "$path" "/usr/local/bin/$name"
        else
            rm -f "/usr/local/bin/$name"
        fi
    done
}
trap cleanup EXIT

backup_cli_wrappers() {
    [[ $EUID -eq 0 ]] || return 0
    local name backup
    for name in randomness-detection randomness-detection-server; do
        if [[ -f "/usr/local/bin/$name" ]]; then
            backup="$(mktemp "${TMPDIR:-/tmp}/rd-cli-backup.XXXXXX")"
            cp -a "/usr/local/bin/$name" "$backup"
            CLI_BACKUPS["$name"]="$backup"
        else
            CLI_BACKUPS["$name"]=""
        fi
    done
}

pick_free_port() {
    python3 -c '
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
'
}

wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    local attempt
    for attempt in $(seq 1 90); do
        if curl -sf "$url" >/dev/null 2>&1; then
            pass "server healthy after ${attempt}s"
            return 0
        fi
        sleep 1
    done
    fail "server did not become healthy within 90s (check ${WORK_DIR}/.run/api.log)"
}

assert_file() {
    [[ -e "$1" ]] || fail "missing expected file: $1"
}

test_gen_api_key() {
    local key
    key="$("$ROOT_DIR/install.sh" --gen-api-key)"
    [[ ${#key} -ge 32 ]] || fail "--gen-api-key length ${#key} < 32"
    pass "--gen-api-key (${#key} chars)"
}

test_help() {
    "$ROOT_DIR/install.sh" --help | grep -q "QUICK START" || fail "--help missing QUICK START"
    pass "--help"
}

prepare_workdir() {
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rd-install-smoke.XXXXXX")"
    log "Work directory: ${WORK_DIR}"

    rsync -a \
        --exclude='.venv/' \
        --exclude='.run/' \
        --exclude='.env' \
        --exclude='.cache/' \
        --exclude='.cache_*/' \
        --exclude='.benchmark_tools/' \
        --exclude='__pycache__/' \
        --exclude='*.egg-info/' \
        --exclude='*.log' \
        --exclude='*.db' \
        "$ROOT_DIR/" "$WORK_DIR/"

    if [[ -d "$ROOT_DIR/.cache" && -f "$ROOT_DIR/.cache/ensemble.pkl" ]]; then
        cp -a "$ROOT_DIR/.cache" "$WORK_DIR/.cache"
        log "Seeded trained model cache (bootstrap skipped in install)"
    else
        log "No local model cache — install will download corpora and train (~2–3 min)"
    fi

    chmod +x "$WORK_DIR/install.sh"
}

run_install() {
    log "Running ./install.sh --no-start (venv + deps + model training)..."
    if ! (cd "$WORK_DIR" && RANDOMNESS_SKIP_SHELL_CONFIG=1 ./install.sh --no-start --host 127.0.0.1 --port "$PORT"); then
        fail "install.sh --no-start failed"
    fi

    assert_file "$WORK_DIR/.venv/bin/python"
    assert_file "$WORK_DIR/.env"
    [[ "$(stat -c '%a' "$WORK_DIR/.env")" == "600" ]] || fail ".env permissions not 600"
    pass "install.sh --no-start completed"
}

run_data_checks() {
    local mode="$1"
    "$WORK_DIR/.venv/bin/python" "$WORK_DIR/test_install_smoke_data.py" \
        --work-dir "$WORK_DIR" \
        --port "$PORT" \
        --mode "$mode"
}

start_server() {
    log "Starting API server on 127.0.0.1:${PORT}..."
    if ! (cd "$WORK_DIR" && RANDOMNESS_SKIP_SHELL_CONFIG=1 ./install.sh --manual --host 127.0.0.1 --port "$PORT"); then
        if [[ -f "$WORK_DIR/.run/api.log" ]]; then
            log "Last 30 lines of api.log:"
            tail -30 "$WORK_DIR/.run/api.log" >&2 || true
        fi
        fail "install.sh --manual failed to start server"
    fi
    wait_for_health
}

stop_server() {
    (cd "$WORK_DIR" && ./install.sh --stop)
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        fail "server still responding after --stop"
    fi
    pass "--stop"
}

main() {
    log "Randomness Detection — install.sh smoke test"
    log "Source: ${ROOT_DIR}"

    command -v curl >/dev/null 2>&1 || fail "curl is required"
    command -v rsync >/dev/null 2>&1 || fail "rsync is required"
    command -v python3 >/dev/null 2>&1 || fail "python3 is required"
    assert_file "$ROOT_DIR/install.sh"
    assert_file "$ROOT_DIR/test_install_smoke_data.py"

    backup_cli_wrappers

    test_gen_api_key
    test_help

    prepare_workdir
    PORT="$(pick_free_port)"
    log "Using port ${PORT}"

    run_install
    run_data_checks artifacts
    run_data_checks cli

    start_server
    run_data_checks api

    stop_server

    log "================================================================"
    log "OVERALL: PASS"
    log "  install + model artifacts"
    log "  CLI: 4 natural + 3 random samples"
    log "  API: /health, /score, /score/batch"
    log "================================================================"
}

main "$@"
