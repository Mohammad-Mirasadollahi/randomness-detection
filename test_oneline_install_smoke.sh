#!/usr/bin/env bash
# Smoke test for the one-line install command:
#   curl -fsSL https://raw.githubusercontent.com/Mohammad-Mirasadollahi/randomness-detection/main/install.sh | bash
#
# Locally simulates the pipe with a file:// git clone (no GitHub required).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO=""
WORK_DIR=""
PORT=""
declare -A CLI_BACKUPS=()

log() { printf '[oneline-smoke] %s\n' "$*" >&2; }
fail() { printf '[oneline-smoke] FAIL: %s\n' "$*" >&2; exit 1; }
pass() { printf '[oneline-smoke] PASS: %s\n' "$*" >&2; }

cleanup() {
    if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
        if [[ -f "$WORK_DIR/.run/api.pid" ]]; then
            (cd "$WORK_DIR" && ./install.sh --stop >/dev/null 2>&1) || true
        fi
        rm -rf "$WORK_DIR"
    fi
    [[ -n "$SOURCE_REPO" && -d "$SOURCE_REPO" ]] && rm -rf "$SOURCE_REPO"
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
    if [[ -f "$WORK_DIR/.run/api.log" ]]; then
        log "Last 30 lines of api.log:"
        tail -30 "$WORK_DIR/.run/api.log" >&2 || true
    fi
    fail "server did not become healthy within 90s"
}

prepare_source_repo() {
    SOURCE_REPO="$(mktemp -d "${TMPDIR:-/tmp}/rd-oneline-source.XXXXXX")"
    log "Building local git source at ${SOURCE_REPO}"

    rsync -a \
        --exclude='.venv/' \
        --exclude='.run/' \
        --exclude='.env' \
        --exclude='.git/' \
        --exclude='.benchmark_tools/' \
        --exclude='__pycache__/' \
        --exclude='*.egg-info/' \
        --exclude='*.log' \
        --exclude='*.db' \
        --exclude='.cache_smoke_train/' \
        "$ROOT_DIR/" "$SOURCE_REPO/"

    if [[ -d "$ROOT_DIR/.cache" && -f "$ROOT_DIR/.cache/ensemble.pkl" ]]; then
        cp -a "$ROOT_DIR/.cache" "$SOURCE_REPO/.cache"
        log "Included trained .cache in source repo (fast install)"
    else
        log "No local model cache — one-line install will train (~2–3 min)"
    fi

    (
        cd "$SOURCE_REPO"
        git init -q
        git add -A
        if [[ -d .cache ]]; then
            git add -f .cache
        fi
        git -c user.email=smoke@test -c user.name=smoke commit -q -m "oneline smoke test source"
    )
    pass "local git source repo ready (file://${SOURCE_REPO})"
}

run_oneline_install() {
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rd-oneline-install.XXXXXX")"
    log "Simulating: curl .../install.sh | bash"
    log "  RANDOMNESS_REPO_URL=file://${SOURCE_REPO}"
    log "  RANDOMNESS_INSTALL_DIR=${WORK_DIR}"

    (
        cd /tmp
        export RANDOMNESS_REPO_URL="file://${SOURCE_REPO}"
        export RANDOMNESS_INSTALL_DIR="${WORK_DIR}"
        export RANDOMNESS_SKIP_SHELL_CONFIG=1
        # Same as: curl -fsSL .../install.sh | bash -s -- --no-start
        bash -s -- --no-start --port "${PORT}" \
            < "${ROOT_DIR}/install.sh"
    )

    [[ -f "${WORK_DIR}/pyproject.toml" ]] || fail "clone/install missing pyproject.toml"
    [[ -x "${WORK_DIR}/.venv/bin/python" ]] || fail "clone/install missing venv"
    [[ -f "${WORK_DIR}/.env" ]] || fail "clone/install missing .env"
    [[ -f "${WORK_DIR}/install.sh" ]] || fail "clone/install missing install.sh"
    pass "one-line pipe install completed in ${WORK_DIR}"
}

verify_network_defaults() {
    grep -q '^RANDOMNESS_HOST=0.0.0.0' "$WORK_DIR/.env" \
        || fail ".env missing RANDOMNESS_HOST=0.0.0.0"
    pass "network default (0.0.0.0)"
}

run_data_checks() {
    local mode="$1"
    "$WORK_DIR/.venv/bin/python" "$WORK_DIR/test_install_smoke_data.py" \
        --work-dir "$WORK_DIR" \
        --port "$PORT" \
        --mode "$mode"
}

start_server() {
    log "Starting server via cloned install.sh --manual ..."
    if ! (cd "$WORK_DIR" && RANDOMNESS_SKIP_SHELL_CONFIG=1 ./install.sh --manual --host 127.0.0.1 --port "$PORT"); then
        fail "install.sh --manual failed in cloned directory"
    fi
    wait_for_health
}

stop_server() {
    (cd "$WORK_DIR" && ./install.sh --stop)
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        fail "server still up after --stop"
    fi
    pass "--stop"
}

main() {
    log "One-line install smoke test"
    log "Project: ${ROOT_DIR}"

    command -v git >/dev/null 2>&1 || fail "git is required"
    command -v curl >/dev/null 2>&1 || fail "curl is required"
    command -v rsync >/dev/null 2>&1 || fail "rsync is required"
    command -v python3 >/dev/null 2>&1 || fail "python3 is required"
    [[ -f "${ROOT_DIR}/install.sh" ]] || fail "install.sh not found"
    [[ -f "${ROOT_DIR}/test_install_smoke_data.py" ]] || fail "test_install_smoke_data.py not found"

    backup_cli_wrappers
    PORT="$(pick_free_port)"
    log "Using port ${PORT}"

    prepare_source_repo
    run_oneline_install
    verify_network_defaults
    run_data_checks artifacts
    run_data_checks cli
    start_server
    run_data_checks api
    stop_server

    log "================================================================"
    log "OVERALL: PASS — one-line install works"
    log "  pipe + git clone + install + train"
    log "  network default (0.0.0.0)"
    log "  CLI + API scoring with real samples"
    log "================================================================"
}

main "$@"
