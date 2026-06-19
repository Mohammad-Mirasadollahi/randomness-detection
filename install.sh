#!/usr/bin/env bash
# Randomness Detection — automated install, bootstrap, and server startup.
set -euo pipefail

# One-line install:
#   curl -fsSL https://raw.githubusercontent.com/Mohammad-Mirasadollahi/randomness-detection/main/install.sh | bash
# Override clone target:
#   RANDOMNESS_REPO_URL=https://github.com/Mohammad-Mirasadollahi/randomness-detection.git bash
#   RANDOMNESS_INSTALL_DIR=$HOME/randomness_detection bash
DEFAULT_REPO_URL="${RANDOMNESS_REPO_URL:-https://github.com/Mohammad-Mirasadollahi/randomness-detection.git}"
DEFAULT_INSTALL_DIR="${RANDOMNESS_INSTALL_DIR:-${HOME}/randomness_detection}"

ensure_project_root() {
    local script="${BASH_SOURCE[0]}"
    local candidate=""

    if [[ -n "${RANDOMNESS_ROOT:-}" && -f "${RANDOMNESS_ROOT}/pyproject.toml" ]]; then
        ROOT_DIR="$(cd "${RANDOMNESS_ROOT}" && pwd)"
        return 0
    fi

    if [[ -f "$script" ]]; then
        candidate="$(cd "$(dirname "$script")" && pwd)"
        if [[ -f "${candidate}/pyproject.toml" ]]; then
            ROOT_DIR="$candidate"
            return 0
        fi
    fi

    command -v git >/dev/null 2>&1 || {
        echo "[install] ERROR: git is required for one-line install." >&2
        exit 1
    }

    echo "[install] One-line install — cloning ${DEFAULT_REPO_URL}" >&2
    echo "[install] Install directory: ${DEFAULT_INSTALL_DIR}" >&2

    if [[ -d "${DEFAULT_INSTALL_DIR}/.git" ]]; then
        git -C "${DEFAULT_INSTALL_DIR}" pull --ff-only
    else
        git clone "${DEFAULT_REPO_URL}" "${DEFAULT_INSTALL_DIR}"
    fi

    export RANDOMNESS_ROOT="${DEFAULT_INSTALL_DIR}"
    exec "${DEFAULT_INSTALL_DIR}/install.sh" --manual "$@"
}

ensure_project_root "$@"
VENV_DIR="${ROOT_DIR}/.venv"
ENV_FILE="${ROOT_DIR}/.env"
RUN_DIR="${ROOT_DIR}/.run"
PID_FILE="${RUN_DIR}/api.pid"
LOG_FILE="${RUN_DIR}/api.log"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
SERVICE_NAME="randomness-detection"
UNIT_FILE_SYSTEM="/etc/systemd/system/${SERVICE_NAME}.service"
UNIT_FILE_USER="${HOME}/.config/systemd/user/${SERVICE_NAME}.service"

DEFAULT_PORT="8765"
HOST="0.0.0.0"
PORT="$DEFAULT_PORT"
DO_START=1
FOREGROUND=0
DO_BOOTSTRAP=1
FORCE_BOOTSTRAP=0
STOP_ONLY=0
STATUS_ONLY=0
RUN_MODE="manual"  # default; override with --systemd
SHOW_GUIDE=0
API_KEY_MODE=""  # print | write | rotate
API_KEY_FORCE=0

usage_guide() {
    cat <<'EOF'
================================================================================
 Randomness Detection — install.sh guide
================================================================================

WHAT THIS SCRIPT DOES
  1. Creates Python virtualenv (.venv)
  2. Installs dependencies from requirements.txt
  3. Generates .env with API key and tuned defaults (if missing)
  4. Downloads public word lists and trains the ML model (first run)
  5. Starts the API on 0.0.0.0 (all network interfaces)
  6. Configures your shell to load .env automatically (no source .env needed)
  7. Verifies health, API scoring, and CLI — ready to use immediately

--------------------------------------------------------------------------------
 ONE-LINE INSTALL (recommended)
--------------------------------------------------------------------------------

  curl -fsSL https://raw.githubusercontent.com/Mohammad-Mirasadollahi/randomness-detection/main/install.sh | bash

  Custom install directory:

    RANDOMNESS_INSTALL_DIR=$HOME/randomness_detection \
      curl -fsSL https://raw.githubusercontent.com/Mohammad-Mirasadollahi/randomness-detection/main/install.sh | bash

  Already cloned? Single command (no prompts):

    ./install.sh

  Restrict network access later in .env (then restart):

    RANDOMNESS_HOST=127.0.0.1              # localhost only
    RANDOMNESS_ALLOWED_HOSTS=api.example.com

--------------------------------------------------------------------------------
 QUICK START (local clone)
--------------------------------------------------------------------------------

  cd randomness_detection
  ./install.sh

  Everything is ready — no source .env, no venv activate.
  Try immediately:

    randomness-detection "hello"
    curl -s http://127.0.0.1:8765/health

--------------------------------------------------------------------------------
 API KEY — standard commands
--------------------------------------------------------------------------------

  Generate a key (print only, for scripts):

    ./install.sh --gen-api-key
    export RANDOMNESS_API_KEY="$(./install.sh --gen-api-key)"

  Generate and save to .env (creates .env if missing):

    ./install.sh --write-api-key

  Replace an existing key (invalidates old clients):

    ./install.sh --rotate-api-key
    ./install.sh --stop && ./install.sh --manual    # restart manual server
    sudo systemctl restart randomness-detection          # restart systemd

  Key format:  secrets.token_urlsafe(48)  — 64 URL-safe chars, min 32 required

  After any key change, restart the server so it picks up the new value.

  Interactive API docs:  http://127.0.0.1:8765/docs

--------------------------------------------------------------------------------
 RUN MODES
--------------------------------------------------------------------------------

  systemd
    - Installs randomness-detection.service
    - Enables auto-start on boot
    - System-wide unit needs root:  sudo ./install.sh --systemd
    - Manage:
        sudo systemctl status randomness-detection
        sudo systemctl restart randomness-detection
        sudo systemctl stop randomness-detection
        sudo journalctl -u randomness-detection -f

  manual
    - Runs server in background via nohup
    - PID file:  .run/api.pid
    - Log file:  .run/api.log
    - Manage:
        ./install.sh --status
        ./install.sh --stop
        tail -f .run/api.log
    - Foreground (debug):  ./install.sh --manual --foreground

--------------------------------------------------------------------------------
 DAILY COMMANDS
--------------------------------------------------------------------------------

  ./install.sh --status          Show if server is running + health JSON
  ./install.sh --stop            Stop manual server and/or systemd service
  ./install.sh --guide           Show this full guide
  ./install.sh --no-start        Install/bootstrap only, don't start server
  ./install.sh --force-bootstrap Re-train model (even if cache exists)
  ./install.sh --skip-bootstrap  Skip model download/training
  ./install.sh --host 127.0.0.1  Bind address (default: 0.0.0.0)
  ./install.sh --port 9000       Bind port (default: 8765)

  Non-interactive install:
    ./install.sh --systemd       Skip prompt, use systemd
    ./install.sh --manual        Skip prompt, use manual background

--------------------------------------------------------------------------------
 API EXAMPLES  (run: source .env)
--------------------------------------------------------------------------------

  # Health check (no API key required)
  curl -s http://127.0.0.1:8765/health

  # Score a single string (1 = natural word, 100 = random)
  curl -s -X POST http://127.0.0.1:8765/score \
    -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"text":"arboraceous"}'

  # Score a random-looking string
  curl -s -X POST http://127.0.0.1:8765/score \
    -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"text":"xK9#mQ2pL"}'

  # Batch score (up to 500 strings)
  curl -s -X POST http://127.0.0.1:8765/score/batch \
    -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"texts":["hello","xK9mQ2","https://example.com"]}'

  # Add exclusion rules (skip scoring for matching strings)
  curl -s -X POST http://127.0.0.1:8765/exclude \
    -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"rules":[{"pattern":"blocked.com","rule_type":"domain"}]}'

  # Check if a string would be excluded
  curl -s -X POST http://127.0.0.1:8765/exclude/check \
    -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"text":"sub.blocked.com"}'

  # Exclusion / cache statistics
  curl -s http://127.0.0.1:8765/exclude/stats \
    -H "Authorization: Bearer $RANDOMNESS_API_KEY"

  # Remove exclusion rules
  curl -s -X DELETE http://127.0.0.1:8765/exclude \
    -H "Authorization: Bearer $RANDOMNESS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"patterns":["blocked.com"]}'

--------------------------------------------------------------------------------
 CLI EXAMPLES  (no API server needed)
--------------------------------------------------------------------------------

  After install, the `randomness-detection` command is on your PATH and works from
  any directory (it loads .env for the model cache and settings automatically):

  # Score from command line
  randomness-detection "hello"

  # JSON output
  randomness-detection "xK9#mQ2" --json

  # Score many words from a file
  randomness-detection -f words.txt

  # Manual bootstrap / re-train
  randomness-detection --bootstrap

  # Run the API server directly (foreground)
  randomness-detection-server --host 127.0.0.1 --port 8765

  Equivalent without the PATH command (from the project dir):
  PYTHONPATH=. .venv/bin/python -m randomness_detection "hello"

--------------------------------------------------------------------------------
 CONFIGURATION  (.env)
--------------------------------------------------------------------------------

  Key variables (auto-generated on first install):

    RANDOMNESS_API_KEY=...          Secret for API auth (keep private)
    RANDOMNESS_HOST=0.0.0.0         Bind address (all interfaces by default)
    RANDOMNESS_ALLOWED_HOSTS=        Optional Host-header allowlist (comma-separated)

  Standard API key commands:

    ./install.sh --gen-api-key        Print a new key (stdout only)
    ./install.sh --write-api-key      Generate + save to .env
    ./install.sh --rotate-api-key     Replace key in .env (restart required)
    RANDOMNESS_PORT=8765            Listen port
    RANDOMNESS_CACHE_DIR=.cache     Model + exclude DB location
    RANDOMNESS_PARALLEL_BACKEND=hybrid   process | thread | hybrid
    RANDOMNESS_INFERENCE_WORKERS=24 CPU workers for scoring
    RANDOMNESS_EXCLUDE_ENABLED=true Enable exclusion fast-path
    RANDOMNESS_SKIP_SCORE_THRESHOLD=30  Skip re-scoring cached low scores

  After editing .env, restart the server:
    ./install.sh --stop && ./install.sh --manual
    # or for systemd:
    sudo systemctl restart randomness-detection

--------------------------------------------------------------------------------
 FILES & DIRECTORIES
--------------------------------------------------------------------------------

  .venv/              Python virtualenv (created by install.sh)
  .env                API key and settings (chmod 600, do not commit)
  .cache/             Trained model, corpus, exclude DB
  .run/api.pid        Manual-mode process ID
  .run/api.log        Manual-mode server log
  requirements.txt    Python dependencies
  Docs/               Full documentation (API, exclusion, testing, …)

--------------------------------------------------------------------------------
 TROUBLESHOOTING
--------------------------------------------------------------------------------

  Server won't start
    tail -50 .run/api.log
    ./install.sh --status

  Port already in use
    ./install.sh --stop
    # or change port:  ./install.sh --port 9000

  Model missing / corrupt
    ./install.sh --force-bootstrap --no-start
    ./install.sh --manual

  Permission denied on systemd
    Use:  sudo ./install.sh --systemd
    Or choose manual mode (option 2)

  Re-run install safely
    ./install.sh is idempotent — existing .venv, .env, and model are reused

  Run integration tests
    PYTHONPATH=. .venv/bin/python run_real_tests.py

--------------------------------------------------------------------------------
 ALL OPTIONS
--------------------------------------------------------------------------------

  --host HOST          Bind host (default: 0.0.0.0)
  --port PORT          Bind port (default: 8765)
  --systemd            Install/start as systemd service (skip prompt)
  --manual             Start as manual background process (skip prompt)
  --foreground         Manual mode: run in foreground (Ctrl+C to stop)
  --no-start           Install/bootstrap only, don't start server
  --force-bootstrap    Re-run training even if model exists
  --skip-bootstrap     Skip bootstrap step
  --stop               Stop running server (manual and/or systemd)
  --status             Show server status
  --gen-api-key        Print a new API key to stdout (script-friendly)
  --write-api-key      Generate API key and save to .env
  --rotate-api-key     Replace API key in .env (use --force to skip prompt)
  --force              With --rotate-api-key: replace without confirmation
  --guide              Show this guide
  -h, --help           Same as --guide

================================================================================
EOF
}

usage() {
    usage_guide
}

log() {
    printf '[install] %s\n' "$*" >&2
}

error() {
    printf '[install] ERROR: %s\n' "$*" >&2
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host)
                HOST="${2:?missing host}"
                shift 2
                ;;
            --port)
                PORT="${2:?missing port}"
                shift 2
                ;;
            --systemd)
                RUN_MODE="systemd"
                shift
                ;;
            --manual)
                RUN_MODE="manual"
                shift
                ;;
            --foreground)
                FOREGROUND=1
                RUN_MODE="manual"
                shift
                ;;
            --no-start)
                DO_START=0
                shift
                ;;
            --force-bootstrap)
                FORCE_BOOTSTRAP=1
                shift
                ;;
            --skip-bootstrap)
                DO_BOOTSTRAP=0
                shift
                ;;
            --stop)
                STOP_ONLY=1
                shift
                ;;
            --status)
                STATUS_ONLY=1
                shift
                ;;
            --guide)
                SHOW_GUIDE=1
                shift
                ;;
            --gen-api-key)
                API_KEY_MODE="print"
                DO_START=0
                DO_BOOTSTRAP=0
                shift
                ;;
            --write-api-key)
                API_KEY_MODE="write"
                DO_START=0
                DO_BOOTSTRAP=0
                shift
                ;;
            --rotate-api-key)
                API_KEY_MODE="rotate"
                DO_START=0
                DO_BOOTSTRAP=0
                shift
                ;;
            --force)
                API_KEY_FORCE=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
}

require_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        error "python3 not found. Install Python 3.10+ first."
    fi

    PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    PY_MAJOR="${PY_VERSION%%.*}"
    PY_MINOR="${PY_VERSION#*.}"
    if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
        error "Python 3.10+ required (found ${PY_VERSION})"
    fi
    log "Python ${PY_VERSION} OK"
}

create_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtualenv at ${VENV_DIR}"
        python3 -m venv "$VENV_DIR"
    else
        log "Virtualenv already exists"
    fi

    if [[ ! -x "$PYTHON_BIN" ]]; then
        error "Virtualenv is broken. Remove .venv and re-run install.sh"
    fi
}

install_deps() {
    log "Installing dependencies..."
    # Use `python -m pip` (not the pip console script) so it keeps working even if
    # the venv was created at a different path and pip's shebang is stale.
    "$PYTHON_BIN" -m pip install --upgrade pip wheel >/dev/null
    "$PYTHON_BIN" -m pip install -r "${ROOT_DIR}/requirements.txt"
    log "Installing randomness-detection package (editable, provides CLI)..."
    "$PYTHON_BIN" -m pip install -e "${ROOT_DIR}" >/dev/null
    log "Dependencies installed"
}

# Directory on PATH to expose the CLI from. Prefer system-wide for root, else user.
cli_bin_dir() {
    if [[ $EUID -eq 0 ]] || [[ -w "/usr/local/bin" ]]; then
        echo "/usr/local/bin"
    else
        echo "${HOME}/.local/bin"
    fi
}

# Install thin launchers on PATH so `randomness-detection` works from any directory,
# without activating the venv. Each launcher loads .env (cache dir, workers, etc.)
# then execs the venv console script.
install_cli_command() {
    local bin_dir target
    bin_dir="$(cli_bin_dir)"
    mkdir -p "$bin_dir"

    for name in randomness-detection randomness-detection-server; do
        target="${bin_dir}/${name}"
        cat >"$target" <<EOF
#!/usr/bin/env bash
# Auto-generated by randomness_detection install.sh — do not edit.
# Loads project settings from .env, then runs the venv console script.
set -a
[ -f "${ENV_FILE}" ] && . "${ENV_FILE}"
set +a
exec "${VENV_DIR}/bin/${name}" "\$@"
EOF
        chmod +x "$target"
    done

    log "CLI installed on PATH: ${bin_dir}/randomness-detection (and -server)"
    case ":${PATH}:" in
        *":${bin_dir}:"*) : ;;
        *)
            log "NOTE: ${bin_dir} is not on your PATH yet. Add it with:"
            log "  echo 'export PATH=\"${bin_dir}:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
            ;;
    esac
}

cpu_count() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    else
        python3 -c 'import os; print(os.cpu_count() or 1)'
    fi
}

python_for_secrets() {
    if [[ -x "$PYTHON_BIN" ]]; then
        echo "$PYTHON_BIN"
    else
        echo "python3"
    fi
}

generate_api_key() {
    "$(python_for_secrets)" -c 'import secrets; print(secrets.token_urlsafe(48))'
}

is_weak_api_key() {
    local key="${1:-}"
    [[ -z "$key" ]] && return 0
    [[ ${#key} -lt 32 ]] && return 0
    [[ "$key" == change-me* ]] && return 0
    return 1
}

read_env_api_key() {
    if [[ ! -f "$ENV_FILE" ]]; then
        return 1
    fi
    local line
    line="$(grep -m1 '^RANDOMNESS_API_KEY=' "$ENV_FILE" 2>/dev/null || true)"
    [[ -n "$line" ]] || return 1
    printf '%s' "${line#RANDOMNESS_API_KEY=}"
}

write_env_file_with_key() {
    local api_key="$1"
    local cpus workers
    cpus="$(cpu_count)"
    workers=$(( cpus < 24 ? cpus : 24 ))
    if [[ "$workers" -lt 1 ]]; then workers=1; fi

    log "Creating ${ENV_FILE}"
    cat >"$ENV_FILE" <<EOF
# =============================================================================
# Randomness Detection — environment configuration (generated by install.sh)
# =============================================================================
#
# QUICK START
# -----------
#   install.sh configures everything automatically — no source .env needed.
#   Open a new terminal (or run: . ~/.config/randomness_detection/env.sh)
#
#   randomness-detection "hello"
#   curl -s http://127.0.0.1:${PORT}/health
#   curl -s -X POST http://127.0.0.1:${PORT}/score \\
#     -H "Authorization: Bearer \$RANDOMNESS_API_KEY" \\
#     -H "Content-Type: application/json" \\
#     -d '{"text":"hello"}'
#
# AFTER CHANGING ANY VALUE BELOW — restart the server:
#   ./install.sh --stop && ./install.sh
#
# SECURITY: chmod 600, never commit to git. Rotate key: ./install.sh --rotate-api-key
# =============================================================================

# --- Authentication (required) ---
# Min 32 chars. Header: Authorization: Bearer <key>  OR  X-API-Key: <key>
RANDOMNESS_API_KEY=${api_key}

# --- Network / API server ---
# 0.0.0.0 = all interfaces | 127.0.0.1 = localhost only
RANDOMNESS_HOST=${HOST}
RANDOMNESS_PORT=${PORT}
# Optional Host allowlist: RANDOMNESS_ALLOWED_HOSTS=api.example.com,localhost

# --- Model cache ---
RANDOMNESS_CACHE_DIR=${ROOT_DIR}/.cache

# --- Parallel scoring ---
RANDOMNESS_PARALLEL_BACKEND=hybrid
RANDOMNESS_INFERENCE_WORKERS=${workers}
RANDOMNESS_INFERENCE_THREADS=${workers}
RANDOMNESS_UVICORN_WORKERS=1

# --- Exclusion & score cache ---
RANDOMNESS_EXCLUDE_ENABLED=true
RANDOMNESS_SKIP_CACHE_ENABLED=true
RANDOMNESS_SKIP_SCORE_THRESHOLD=30

# --- Internal ---
PYTHONPATH=${ROOT_DIR}
EOF
    chmod 600 "$ENV_FILE"
}

set_api_key_in_env() {
    local api_key="$1"
    if [[ ! -f "$ENV_FILE" ]]; then
        write_env_file_with_key "$api_key"
        return
    fi

    if grep -q '^RANDOMNESS_API_KEY=' "$ENV_FILE"; then
        sed -i "s|^RANDOMNESS_API_KEY=.*|RANDOMNESS_API_KEY=${api_key}|" "$ENV_FILE"
    else
        echo "RANDOMNESS_API_KEY=${api_key}" >>"$ENV_FILE"
    fi
    chmod 600 "$ENV_FILE"
}

print_api_key_restart_hint() {
    if systemd_active 2>/dev/null; then
        log "Restart systemd service:  sudo systemctl restart ${SERVICE_NAME}"
    elif server_running 2>/dev/null; then
        log "Restart manual server:   ./install.sh --stop && ./install.sh"
    else
        log "Start server when ready: ./install.sh"
    fi
    log "Settings reload in new shells automatically (or run: . ~/.config/randomness_detection/env.sh)"
}

handle_api_key_command() {
    require_python
    create_venv

    local api_key current_key

    case "$API_KEY_MODE" in
        print)
            if ! command -v python3 >/dev/null 2>&1; then
                error "python3 not found. Install Python 3.10+ first."
            fi
            python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
            exit 0
            ;;
        write)
            api_key="$(generate_api_key)"
            if [[ -f "$ENV_FILE" ]]; then
                current_key="$(read_env_api_key || true)"
                if [[ -n "$current_key" ]] && ! is_weak_api_key "$current_key"; then
                    log "Valid API key already exists in ${ENV_FILE} — not overwriting"
                    log "Use --rotate-api-key to replace it"
                    exit 0
                fi
                log "Updating weak or missing API key in ${ENV_FILE}"
                set_api_key_in_env "$api_key"
            else
                log "Writing new API key to ${ENV_FILE}"
                write_env_file_with_key "$api_key"
            fi
            log "API key saved (chmod 600)"
            print_api_key_restart_hint
            exit 0
            ;;
        rotate)
            api_key="$(generate_api_key)"
            if [[ ! -f "$ENV_FILE" ]]; then
                log "No ${ENV_FILE} found — creating with new API key"
                write_env_file_with_key "$api_key"
                log "API key saved (chmod 600)"
                print_api_key_restart_hint
                exit 0
            fi

            current_key="$(read_env_api_key || true)"
            if [[ -n "$current_key" ]] && ! is_weak_api_key "$current_key" && [[ "$API_KEY_FORCE" -ne 1 ]]; then
                if [[ -t 0 ]]; then
                    echo "" >&2
                    echo "This will replace the existing API key in ${ENV_FILE}." >&2
                    echo "All clients using the old key will lose access." >&2
                    local confirm
                    read -rp "Continue? [y/N]: " confirm
                    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                        log "Cancelled"
                        exit 0
                    fi
                else
                    error "Refusing to rotate without confirmation in non-interactive mode. Use --force."
                fi
            fi

            set_api_key_in_env "$api_key"
            log "API key rotated in ${ENV_FILE}"
            print_api_key_restart_hint
            exit 0
            ;;
        *)
            error "Unknown API key command"
            ;;
    esac
}

write_env_file() {
    if [[ -f "$ENV_FILE" ]]; then
        # Keep existing secrets; refresh host/port if .env was just created elsewhere
        if grep -q '^RANDOMNESS_HOST=' "$ENV_FILE"; then
            sed -i "s|^RANDOMNESS_HOST=.*|RANDOMNESS_HOST=${HOST}|" "$ENV_FILE"
        else
            echo "RANDOMNESS_HOST=${HOST}" >>"$ENV_FILE"
        fi
        if grep -q '^RANDOMNESS_PORT=' "$ENV_FILE"; then
            sed -i "s|^RANDOMNESS_PORT=.*|RANDOMNESS_PORT=${PORT}|" "$ENV_FILE"
        else
            echo "RANDOMNESS_PORT=${PORT}" >>"$ENV_FILE"
        fi
        log "Using existing ${ENV_FILE}"
        return
    fi

    local cpus workers api_key
    cpus="$(cpu_count)"
    workers=$(( cpus < 24 ? cpus : 24 ))
    if [[ "$workers" -lt 1 ]]; then workers=1; fi

    api_key="$(generate_api_key)"

    write_env_file_with_key "$api_key"
}

load_env() {
    # shellcheck disable=SC1090
    set -a
    source "$ENV_FILE"
    set +a
    export PYTHONPATH="${ROOT_DIR}"
    HOST="${RANDOMNESS_HOST:-$HOST}"
    PORT="${RANDOMNESS_PORT:-$PORT}"
}

configure_shell_env() {
    if [[ -n "${RANDOMNESS_SKIP_SHELL_CONFIG:-}" ]]; then
        log "Shell auto-config skipped (RANDOMNESS_SKIP_SHELL_CONFIG)"
        return 0
    fi

    local config_dir="${HOME}/.config/randomness_detection"
    local env_sh="${config_dir}/env.sh"
    local marker="# randomness_detection (auto-configured by install.sh)"
    local bin_dir profile block

    bin_dir="$(cli_bin_dir)"
    mkdir -p "$config_dir"

    cat >"$env_sh" <<EOF
# Auto-generated by install.sh — loads settings in every new shell.
# Re-run ./install.sh to refresh after moving the install directory.
set -a
[ -f "${ENV_FILE}" ] && . "${ENV_FILE}"
set +a
case ":\${PATH}:" in
    *":${bin_dir}:"*) ;;
    *) export PATH="${bin_dir}:\${PATH}" ;;
esac
EOF
    chmod 600 "$env_sh"

    block=$(cat <<EOF

${marker}
[ -f "${env_sh}" ] && . "${env_sh}"
EOF
)

    for profile in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
        if [[ ! -f "$profile" ]]; then
            touch "$profile"
        fi
        if grep -qF "$marker" "$profile" 2>/dev/null; then
            log "Shell already configured: ${profile}"
            continue
        fi
        printf '%s\n' "$block" >>"$profile"
        log "Shell auto-configured: ${profile}"
    done

    log "New terminals load .env automatically via ${env_sh}"

    # Apply immediately in the current shell (same install session).
    # shellcheck disable=SC1090
    . "$env_sh"
}

run_post_install_verify() {
    local api_key health score cli_out

    [[ "$DO_START" -eq 0 ]] && return 0
    [[ -f "$ENV_FILE" ]] || return 0

    api_key="$(read_env_api_key || true)"
    [[ -n "$api_key" ]] || return 0

    log "Post-install verification (no manual setup required)..."

    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "  health: OK"
    else
        log "  health: FAILED"
        return 1
    fi

    score="$(curl -sf -X POST "http://127.0.0.1:${PORT}/score" \
        -H "Authorization: Bearer ${api_key}" \
        -H "Content-Type: application/json" \
        -d '{"text":"hello"}' 2>/dev/null || true)"
    if [[ -n "$score" ]] && [[ "$score" == *'"score"'* ]]; then
        log "  API score: OK"
    else
        log "  API score: FAILED"
        return 1
    fi

    cli_out="$(randomness-detection "hello" 2>/dev/null || true)"
    if [[ "$cli_out" == *"Score:"* ]]; then
        log "  CLI: OK (works without source .env or venv activate)"
    else
        log "  CLI: FAILED"
        return 1
    fi

    log "Post-install verification: PASS"
    return 0
}

model_exists() {
    local cache_dir="${RANDOMNESS_CACHE_DIR:-${ROOT_DIR}/.cache}"
    [[ -f "${cache_dir}/ensemble.pkl" && -f "${cache_dir}/english.freq" && -f "${cache_dir}/metadata.json" ]]
}

run_bootstrap() {
    if [[ "$DO_BOOTSTRAP" -eq 0 ]]; then
        log "Bootstrap skipped"
        return
    fi

    local bootstrap_args=()
    bootstrap_args+=(--cache-dir "${RANDOMNESS_CACHE_DIR}")

    if [[ "$FORCE_BOOTSTRAP" -eq 1 ]]; then
        bootstrap_args+=(--force)
    elif model_exists; then
        log "Model already exists, skipping bootstrap"
        return
    fi

    log "============================================================"
    log " Model training — automatic on first install"
    log "============================================================"
    log " Downloads public English word lists and trains on your CPU."
    log " All processing stays local — nothing is uploaded."
    log " Live progress and elapsed time are shown below."
    log " Typical duration: 1–3 minutes (depends on CPU)."
    log "============================================================"

    local start_ts elapsed
    start_ts=$(date +%s)

    if ! PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m randomness_detection.bootstrap_cli \
        "${bootstrap_args[@]}"; then
        error "Model training failed (see bootstrap logs above)"
    fi

    elapsed=$(( $(date +%s) - start_ts ))
    log "Model training finished (${elapsed}s total)"
}

systemd_scope() {
    if [[ -f "$UNIT_FILE_SYSTEM" ]] || [[ $EUID -eq 0 ]]; then
        echo "system"
    elif [[ -f "$UNIT_FILE_USER" ]]; then
        echo "user"
    elif [[ $EUID -ne 0 ]] && systemctl --user show-environment >/dev/null 2>&1; then
        echo "user"
    else
        echo "system"
    fi
}

systemctl_cmd() {
    if [[ "$(systemd_scope)" == "user" ]]; then
        systemctl --user "$@"
    else
        systemctl "$@"
    fi
}

systemd_unit_path() {
    if [[ "$(systemd_scope)" == "user" ]]; then
        echo "$UNIT_FILE_USER"
    else
        echo "$UNIT_FILE_SYSTEM"
    fi
}

systemd_installed() {
    [[ -f "$(systemd_unit_path)" ]]
}

systemd_active() {
    systemd_installed && systemctl_cmd is-active --quiet "$SERVICE_NAME" 2>/dev/null
}

choose_run_mode() {
    if [[ -n "${RUN_MODE:-}" ]]; then
        return
    fi
    RUN_MODE="manual"
    log "Run mode: manual (use --systemd for a boot service)"
}

server_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_manual_server() {
    if server_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        log "Stopping manual server (PID ${pid})..."
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        log "Manual server stopped"
    fi
}

stop_systemd_service() {
    if systemd_installed; then
        log "Stopping systemd service (${SERVICE_NAME})..."
        systemctl_cmd stop "$SERVICE_NAME" 2>/dev/null || true
        log "Systemd service stopped"
    fi
}

stop_server() {
    stop_manual_server
    stop_systemd_service
    if ! server_running && ! systemd_active; then
        log "Server is not running"
        rm -f "$PID_FILE"
    fi
}

show_status() {
    local any_running=0

    if systemd_installed; then
        log "Systemd unit: $(systemd_unit_path)"
        systemctl_cmd status "$SERVICE_NAME" --no-pager 2>/dev/null || true
        any_running=1
    fi

    if server_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        log "Manual server is running (PID ${pid})"
        log "Log file: ${LOG_FILE}"
        any_running=1
    fi

    if [[ "$any_running" -eq 0 ]]; then
        log "Server is not running"
    elif command -v curl >/dev/null 2>&1; then
        curl -sf "http://127.0.0.1:${PORT}/health" | "$PYTHON_BIN" -m json.tool 2>/dev/null || true
    fi
}

wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    for _ in $(seq 1 60); do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

install_systemd_service() {
    if ! command -v systemctl >/dev/null 2>&1; then
        error "systemd is not available on this system. Choose manual mode instead."
    fi

    local unit_path scope wanted_by
    scope="$(systemd_scope)"
    unit_path="$(systemd_unit_path)"

    if [[ "$scope" == "system" && $EUID -ne 0 ]]; then
        error "Installing a system-wide systemd service requires root. Re-run with sudo or choose manual mode."
    fi

    if [[ "$scope" == "user" ]]; then
        mkdir -p "$(dirname "$unit_path")"
        wanted_by="default.target"
    else
        wanted_by="multi-user.target"
    fi

    stop_manual_server

    log "Installing systemd unit: ${unit_path}"
    cat >"$unit_path" <<EOF
[Unit]
Description=Randomness Detection API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
EnvironmentFile=${ENV_FILE}
Environment=PYTHONPATH=${ROOT_DIR}
ExecStart=${PYTHON_BIN} -m randomness_detection.api_server --host \${RANDOMNESS_HOST} --port \${RANDOMNESS_PORT}
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=${wanted_by}
EOF

    systemctl_cmd daemon-reload
    systemctl_cmd enable "$SERVICE_NAME"
    systemctl_cmd restart "$SERVICE_NAME"

    if wait_for_health; then
        log "Systemd service is up and healthy"
    else
        error "Service failed to start. Check: systemctl status ${SERVICE_NAME}"
    fi
}

start_manual_server() {
    mkdir -p "$RUN_DIR"

    if server_running; then
        log "Manual server already running (PID $(cat "$PID_FILE"))"
        return
    fi

    if systemd_active; then
        log "Systemd service is already running — skipping manual start"
        return
    fi

    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
        sleep 1
    fi

    log "Starting API server manually on ${HOST}:${PORT}..."

    if [[ "$FOREGROUND" -eq 1 ]]; then
        log "Running in foreground (Ctrl+C to stop)"
        exec bash -c "set -a; source '${ENV_FILE}'; set +a; exec '${PYTHON_BIN}' -m randomness_detection.api_server --host '${HOST}' --port '${PORT}'"
    fi

    nohup bash -c "set -a; source '${ENV_FILE}'; set +a; exec '${PYTHON_BIN}' -m randomness_detection.api_server --host '${HOST}' --port '${PORT}'" \
        >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"

    if wait_for_health; then
        log "Manual server is up and healthy"
    else
        error "Server failed to start. Check ${LOG_FILE}"
    fi
}

start_server() {
    if [[ "$DO_START" -eq 0 ]]; then
        log "Server start skipped (--no-start)"
        return
    fi

    choose_run_mode

    if [[ "$RUN_MODE" == "systemd" ]]; then
        install_systemd_service
    else
        start_manual_server
    fi
}

print_summary() {
    cat <<EOF

============================================================
 Randomness Detection — ready
============================================================
 Run mode:    ${RUN_MODE:-manual}
 Listen:      ${HOST}:${PORT}  (0.0.0.0 = all interfaces)
 API URL:     http://${HOST}:${PORT}
 Health:      http://127.0.0.1:${PORT}/health
 Docs:        http://127.0.0.1:${PORT}/docs
 API key:     ${ENV_FILE}  (RANDOMNESS_API_KEY)
 Cache:       ${RANDOMNESS_CACHE_DIR:-${ROOT_DIR}/.cache}

 Restrict access: edit ${ENV_FILE} → RANDOMNESS_HOST=127.0.0.1
EOF

    if [[ "${RUN_MODE}" == "systemd" ]]; then
        cat <<EOF
 Systemd:     ${SERVICE_NAME}.service
 Unit file:   $(systemd_unit_path)

 Manage:
   sudo systemctl status ${SERVICE_NAME}
   sudo systemctl restart ${SERVICE_NAME}
   sudo systemctl stop ${SERVICE_NAME}
   sudo journalctl -u ${SERVICE_NAME} -f
EOF
    else
        cat <<EOF
 Log:         ${LOG_FILE}
 PID:         ${PID_FILE}

 Manage:
   ./install.sh --status
   ./install.sh --stop
   tail -f ${LOG_FILE}
EOF
    fi

    cat <<EOF

------------------------------------------------------------
 READY TO USE (no manual setup)
------------------------------------------------------------

 CLI (works from any directory, .env loaded automatically):

    randomness-detection "hello"
    randomness-detection "xK9#mQ2pL" --json

 Health (no auth):

    curl -s http://127.0.0.1:${PORT}/health

 API score (\$RANDOMNESS_API_KEY is set automatically in new shells):

    curl -s -X POST http://127.0.0.1:${PORT}/score \\
      -H "Authorization: Bearer \$RANDOMNESS_API_KEY" \\
      -H "Content-Type: application/json" \\
      -d '{"text":"hello"}'

 Current shell missing \$RANDOMNESS_API_KEY? Run once:

    . ~/.config/randomness_detection/env.sh

------------------------------------------------------------
 More help:  ./install.sh --guide
 Docs:       ${ROOT_DIR}/Docs/
============================================================
EOF
}

main() {
    parse_args "$@"

    cd "$ROOT_DIR"

    if [[ "$SHOW_GUIDE" -eq 1 ]]; then
        usage_guide
        exit 0
    fi

    if [[ -n "$API_KEY_MODE" ]]; then
        handle_api_key_command
    fi

    if [[ "$STOP_ONLY" -eq 1 ]]; then
        [[ -f "$ENV_FILE" ]] && load_env || true
        stop_server
        exit 0
    fi

    if [[ "$STATUS_ONLY" -eq 1 ]]; then
        [[ -f "$ENV_FILE" ]] && load_env || true
        show_status
        exit 0
    fi

    log "Randomness Detection install — ${ROOT_DIR}"
    if [[ -t 0 ]] && [[ "$DO_START" -eq 1 ]] && [[ -z "$RUN_MODE" ]]; then
        log "Tip: run ./install.sh --guide for the full usage guide"
    fi
    require_python
    create_venv
    install_deps
    write_env_file
    load_env
    install_cli_command
    run_bootstrap
    start_server
    configure_shell_env
    run_post_install_verify || true
    print_summary
}

main "$@"
