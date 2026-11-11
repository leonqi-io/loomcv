#!/bin/bash
# LoomCV start script: double-click it in Finder, or run it in a terminal.
# It checks uv, installs the packages, asks four questions on the first start,
# checks the model server and opens LoomCV in the browser.
# Written for bash 3.2, the version macOS ships.

cd "$(dirname "$0")" || exit 1
# Raise the open-file limit for this run; macOS starts at 256.
ulimit -n 4096 2>/dev/null || true
HERE=$(pwd)
PORT=7860
ADDRESS="http://localhost:$PORT"
LOCAL_ADDRESS="http://127.0.0.1:$PORT/"
DEFAULT_SERVER="http://localhost:11434/v1"
UV_MIN="0.8.0"
CONFIG="config/models.local.yaml"
TEMPLATE="config/models.local.example.yaml"

# uv's usual places, searched after PATH (a double-click starts with a short PATH).
PATH="$PATH:$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/loomcv.XXXXXX") || exit 1
HELPER=""
cleanup() {
    if [ -n "$HELPER" ]; then
        kill "$HELPER" 2>/dev/null
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

# True when the value holds a character that is not safe in the unquoted .env file.
has_bad() {
    case "$1" in
        *[[:space:]]* | *"#"* | *"'"* | *'"'* | *'\'* | *'$'*) return 0 ;;
    esac
    return 1
}

# ask TEXT HIDDEN ALLOW_EMPTY CHECK: prints the question, reads one answer into ANSWER.
ask() {
    local text="$1" hidden="$2" allow_empty="$3" check="$4" raw
    ANSWER=""
    while true; do
        printf '%s\n' "$text"
        if [ "$hidden" = 1 ]; then
            read -r -s -p "> " raw || exit 1
            echo
        else
            read -r -p "> " raw || exit 1
        fi
        raw=$(trim "$raw")
        if [ -z "$raw" ]; then
            if [ "$allow_empty" = 1 ]; then
                return 0
            fi
            echo "Please type a value."
            continue
        fi
        if has_bad "$raw"; then
            echo "Please type a value."
            continue
        fi
        if [ "$check" = url ]; then
            case "$raw" in
                http://* | https://*) ;;
                *)
                    echo "Please type a value."
                    continue
                    ;;
            esac
        fi
        ANSWER="$raw"
        return 0
    done
}

# yes_no DEFAULT PROMPT: DEFAULT is y or n; anything unclear counts as the default.
yes_no() {
    local default="$1" prompt="$2" ans
    read -r -p "$prompt " ans || ans=""
    ans=$(trim "$ans" | tr 'A-Z' 'a-z')
    case "$ans" in
        y | yes) return 0 ;;
        n | no) return 1 ;;
    esac
    [ "$default" = y ]
}

# True when the first dotted version is older than the second.
version_older() {
    local result
    result=$(awk -v a="$1" -v b="$2" 'BEGIN {
        split(a, x, "."); split(b, y, ".");
        for (i = 1; i <= 3; i++) {
            if ((x[i] + 0) < (y[i] + 0)) { print 1; exit }
            if ((x[i] + 0) > (y[i] + 0)) { print 0; exit }
        }
        print 0
    }')
    [ "$result" = 1 ]
}

# The value of KEY in .env, without a trailing carriage return.
env_value() {
    grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '\r'
}

port_in_use() {
    if command -v lsof >/dev/null 2>&1; then
        [ -n "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null)" ]
    else
        curl -s -m 2 -o /dev/null "$LOCAL_ADDRESS"
    fi
}

# ---------------------------------------------------------------- opening

echo "LoomCV"
printf 'Folder: %s\n' "$HERE"

# ---------------------------------------------------------------- uv

if ! command -v uv >/dev/null 2>&1; then
    echo
    cat <<'EOF'
LoomCV needs uv, a free tool that installs Python and LoomCV's packages.
It is installed for your user only, with this command from uv's own website:

    curl -LsSf https://astral.sh/uv/install.sh | sh

EOF
    if yes_no n "Install uv now? [y/N]"; then
        echo "Installing uv…"
        if (
            set -o pipefail
            curl -LsSf https://astral.sh/uv/install.sh | sh
        ); then
            PATH="$HOME/.local/bin:$PATH"
            hash -r
            echo "uv is installed."
        else
            echo "uv could not be installed. Check your internet connection and double-click start.command again."
            exit 1
        fi
    else
        echo "LoomCV can't start without uv. Nothing was changed. Run the command above yourself, then double-click start.command again."
        exit 1
    fi
fi

UV_VERSION=$(uv --version 2>/dev/null | awk '{print $2}')
if version_older "$UV_VERSION" "$UV_MIN"; then
    echo
    printf 'Your uv is version %s. LoomCV needs %s or newer.\n' "$UV_VERSION" "$UV_MIN"
    cat <<'EOF'
Update it with one of these commands, then double-click start.command again:

    uv self update          (if you installed uv with its own installer)
    brew upgrade uv         (if you installed uv with Homebrew)

Nothing was changed.
EOF
    exit 1
fi

# ---------------------------------------------------------------- Rosetta 2

if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    if ! /usr/bin/arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
        echo
        cat <<'EOF'
LoomCV makes its PDFs with a tool that needs Rosetta 2, Apple's free compatibility layer.
It is installed with this command from Apple:

    softwareupdate --install-rosetta --agree-to-license

EOF
        if yes_no n "Install Rosetta 2 now? [y/N]"; then
            echo "Installing Rosetta 2…"
            if softwareupdate --install-rosetta --agree-to-license; then
                echo "Rosetta 2 is installed."
            else
                echo 'Continuing without Rosetta 2. LoomCV will start, but "Make the PDF" will fail until it is installed.'
            fi
        else
            echo 'Continuing without Rosetta 2. LoomCV will start, but "Make the PDF" will fail until it is installed.'
        fi
    fi
fi

# ---------------------------------------------------------------- packages

echo
if [ ! -d .venv ]; then
    echo "Installing LoomCV's packages. This first install downloads about 1.5 GB and can take a while. Later starts skip it."
else
    echo "Checking LoomCV's packages…"
fi
if uv sync --locked --all-groups; then
    echo "Packages are ready."
else
    echo "The packages could not be installed. Check your internet connection and free disk space, then double-click start.command again. The lines above say what went wrong."
    exit 1
fi

# ---------------------------------------------------------------- the questions

if [ ! -f .env ] || [ ! -f "$CONFIG" ]; then
    echo
    cat <<'EOF'
LoomCV needs to know which model to use. Four questions; your answers are saved, so you are asked only once.
If you have not set up a model yet, close this window, follow step 1 of the README, and start again.
EOF
    echo

    ask "$(
        cat <<'EOF'
1/4  Where is your model server?
     Press Enter for Ollama on this computer (http://localhost:11434/v1).
     Any other server, local or hosted: paste its base URL (it usually ends in /v1).
EOF
    )" 0 1 url
    BASE_URL="$ANSWER"
    ADDRESS_GIVEN=1
    if [ -z "$BASE_URL" ]; then
        BASE_URL="$DEFAULT_SERVER"
        ADDRESS_GIVEN=0
    fi

    ask "$(
        cat <<'EOF'
2/4  Which model should write? Type its name exactly as your model server lists it
     (for example: qwen3.8:27b-mlx).
EOF
    )" 0 0 text
    MODEL="$ANSWER"

    ask "$(
        cat <<'EOF'
3/4  API key? Press Enter if your model server needs none. What you type is not shown.
EOF
    )" 1 1 text
    API_KEY="$ANSWER"

    ask "$(
        cat <<'EOF'
4/4  Which embedding model? Type its name exactly as your model server lists it
     (for example: bge-m3).
EOF
    )" 0 0 text
    EMBED_MODEL="$ANSWER"

    EMBED_URL="$BASE_URL"
    EMBED_KEY="$API_KEY"
    if [ "$ADDRESS_GIVEN" = 1 ]; then
        if ! yes_no y "Does that same server also provide the embedding model? [Y/n]"; then
            ask "$(
                cat <<'EOF'
     Where is the embedding server? Press Enter for Ollama on this computer
     (http://localhost:11434/v1), or paste its base URL.
EOF
            )" 0 1 url
            EMBED_URL="${ANSWER:-$DEFAULT_SERVER}"
            ask "$(
                cat <<'EOF'
     API key for the embedding server? Press Enter if it needs none. What you type is not shown.
EOF
            )" 1 1 text
            EMBED_KEY="$ANSWER"
        fi
    fi

    SAVED=1
    umask 077
    mkdir -p config
    if [ ! -f "$TEMPLATE" ]; then
        SAVED=0
    fi
    if [ "$SAVED" = 1 ]; then
        {
            printf 'LOOMCV_BASE_URL=%s\n' "$BASE_URL"
            printf 'LOOMCV_API_KEY=%s\n' "$API_KEY"
            printf 'LOOMCV_EMBED_BASE_URL=%s\n' "$EMBED_URL"
            printf 'LOOMCV_EMBED_API_KEY=%s\n' "$EMBED_KEY"
        } 2>/dev/null >.env || SAVED=0
    fi
    if [ "$SAVED" = 1 ]; then
        TEXT=$(cat "$TEMPLATE")
        TEXT="${TEXT//REPLACE_WITH_MODEL_ID/$MODEL}"
        TEXT="${TEXT//REPLACE_WITH_EMBEDDING_MODEL_ID/$EMBED_MODEL}"
        printf '%s\n' "$TEXT" 2>/dev/null >"$CONFIG" || SAVED=0
    fi
    if [ "$SAVED" = 0 ]; then
        rm -f .env "$CONFIG"
        echo
        printf 'Your answers could not be saved in %s. Check that you are allowed to write there, then start again.\n' "$HERE"
        exit 1
    fi

    # The project's own readers must accept what was written.
    if ! uv run --no-sync --env-file .env python -c '
from pathlib import Path
from loomcv.llm import load_node_config
from loomcv.retrieve.rerank import load_rerank_config
from loomcv.sync.embedding import load_embedding_config

path = Path("config/models.local.yaml")
for node in ("parse_jd", "coverage", "ingest", "rewrite"):
    load_node_config(node, path)
load_embedding_config(path)
load_rerank_config(path)
' >/dev/null 2>&1; then
        rm -f .env "$CONFIG"
        echo
        cat <<'EOF'
Your answers could not be used as LoomCV's settings, so they were not kept: .env and config/models.local.yaml were removed again.
Double-click start.command to answer the questions once more. Check the two model names as you type them: each must be the name exactly as your model server lists it.
EOF
        exit 1
    fi

    echo
    cat <<'EOF'
Saved your answers in this folder: .env and config/models.local.yaml.
To change them later, delete the file .env and double-click start.command again.
EOF
fi

# ---------------------------------------------------------------- the model server

BASE_URL=$(env_value LOOMCV_BASE_URL)
API_KEY=$(env_value LOOMCV_API_KEY)
EMBED_URL=$(env_value LOOMCV_EMBED_BASE_URL)
EMBED_KEY=$(env_value LOOMCV_EMBED_API_KEY)
MODEL=$(sed -n 's/.*model: &model "\([^"]*\)".*/\1/p' "$CONFIG" 2>/dev/null | head -1)
EMBED_MODEL=$(sed -n 's/^  model: "\([^"]*\)".*/\1/p' "$CONFIG" 2>/dev/null | head -1)

# check_server URL KEY OUTFILE: fetches the server's model list; true on HTTP 200.
check_server() {
    local url="${1%/}" key="$2" out="$3" header code
    header="$WORK/header"
    if [ -n "$key" ]; then
        (
            umask 077
            printf 'Authorization: Bearer %s\n' "$key" >"$header"
        )
        code=$(curl -sS -m 8 -o "$out" -w '%{http_code}' -H "@$header" "$url/models" 2>/dev/null)
        rm -f "$header"
    else
        code=$(curl -sS -m 8 -o "$out" -w '%{http_code}' "$url/models" 2>/dev/null)
    fi
    [ "$code" = 200 ]
}

no_answer() {
    echo
    printf 'No answer from %s.\n' "$1"
    cat <<'EOF'
Check three things:
  1. The model server is running (for a local one: its app or its terminal window is open).
  2. The address is right.
  3. The API key is right, if your server needs one.
Then double-click start.command again. To re-enter your answers, delete the file .env in this folder first.
EOF
}

# A name is found when the list has it, or, for a name without a tag, "<name>:latest"
# (Ollama resolves an untagged name to :latest).
model_listed() {
    grep -Fq "\"$1\"" "$2" && return 0
    case "$1" in
        *:*) return 1 ;;
    esac
    grep -Fq "\"${1}:latest\"" "$2"
}

note_missing() {
    if [ -n "$1" ] && ! model_listed "$1" "$2"; then
        printf 'Note: the server'"'"'s list has no model named "%s". If the name is mistyped, LoomCV will start but its first model step will fail. To re-enter it, delete the file .env and start again.\n' "$1"
    fi
}

echo
echo "Checking that the model server answers…"
if ! check_server "$BASE_URL" "$API_KEY" "$WORK/models_main.json"; then
    no_answer "$BASE_URL"
    exit 1
fi
cp "$WORK/models_main.json" "$WORK/models_embed.json"
if [ "${EMBED_URL%/}" != "${BASE_URL%/}" ]; then
    if ! check_server "$EMBED_URL" "$EMBED_KEY" "$WORK/models_embed.json"; then
        no_answer "$EMBED_URL"
        exit 1
    fi
fi
echo "The model server answered."
note_missing "$MODEL" "$WORK/models_main.json"
note_missing "$EMBED_MODEL" "$WORK/models_embed.json"

# ---------------------------------------------------------------- address and privacy

if port_in_use; then
    echo
    echo "Something is already using $ADDRESS — LoomCV may still be running in another window. Close that window and start again, or open that address in your browser."
    exit 1
fi

export GRADIO_SERVER_PORT="$PORT"
export GRADIO_NUM_PORTS=1
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$HERE/.models"
export GRADIO_TEMP_DIR="$HERE/.gradio"
mkdir -p "$GRADIO_TEMP_DIR"

# ---------------------------------------------------------------- start

FIRST_MODEL=0
if [ -z "$(ls -A .models 2>/dev/null)" ]; then
    FIRST_MODEL=1
fi

echo
echo "Starting LoomCV…"

(
    waited=0
    while [ "$waited" -lt 180 ]; do
        if curl -s -m 2 -o /dev/null "$LOCAL_ADDRESS"; then
            touch "$WORK/up"
            printf 'LoomCV is running at %s — opening your browser.\n' "$ADDRESS"
            echo "Keep this window open while you use LoomCV. To stop LoomCV, close this window."
            if [ "$FIRST_MODEL" = 1 ]; then
                echo "The first résumé you build also downloads a ranking model (about 2.2 GB) into the .models folder here. That happens once."
            fi
            if command -v open >/dev/null 2>&1; then
                open "$ADDRESS"
            elif command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$ADDRESS" >/dev/null 2>&1
            fi
            exit 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    touch "$WORK/timeout"
    echo "LoomCV did not start. The lines above say what went wrong. If you can't tell, copy them into a new issue at https://github.com/leonqi-io/loomcv/issues — leave out anything private."
) &
HELPER=$!

uv run --no-sync --env-file .env pipeline app --config "$CONFIG"

if [ -e "$WORK/up" ]; then
    echo
    echo "LoomCV has stopped. Double-click start.command to start it again."
elif [ ! -e "$WORK/timeout" ]; then
    echo
    echo "LoomCV did not start. The lines above say what went wrong. If you can't tell, copy them into a new issue at https://github.com/leonqi-io/loomcv/issues — leave out anything private."
    exit 1
fi
