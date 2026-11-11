#!/bin/bash
# LoomCV uninstall script: double-click it in Finder, or run it in a terminal.
# It removes only what this folder holds, one question at a time, and only after a yes.
# Written for bash 3.2, the version macOS ships.

cd "$(dirname "$0")" || exit 1
HERE=$(pwd)
PORT=7860
LOCAL_ADDRESS="http://127.0.0.1:$PORT/"

trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
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

port_in_use() {
    if command -v lsof >/dev/null 2>&1; then
        [ -n "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null)" ]
    else
        curl -s -m 2 -o /dev/null "$LOCAL_ADDRESS"
    fi
}

# The size of a folder, or "(not present)".
size_of() {
    if [ -e "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "(not present)"
    fi
}

FAILED=0
# Prints the failure sentence when the path is still there after its rm.
still_there() {
    if [ -e "$1" ]; then
        printf 'Could not remove %s. Close anything that has it open and run uninstall.command again.\n' "$1"
        FAILED=1
    fi
}

# Never delete in the wrong folder.
if ! grep -Eq '^name = "loomcv"$' pyproject.toml 2>/dev/null || [ ! -f start.command ]; then
    echo "uninstall.command is not inside the LoomCV folder, so it removed nothing."
    echo "Put it back in the LoomCV folder, next to start.command, and double-click it there."
    exit 1
fi

# ---------------------------------------------------------------- opening

echo "Uninstall LoomCV"
printf 'Folder: %s\n' "$HERE"
echo "Nothing is removed until you answer yes."

if port_in_use; then
    echo
    echo "LoomCV is still running. Close its Terminal window first, then double-click uninstall.command again."
    exit 1
fi

# ---------------------------------------------------------------- what was found

PRESENT=0
for item in .models .venv 02_database 04_applications .gradio .env config/models.local.yaml; do
    if [ -e "$item" ]; then
        PRESENT=1
    fi
done

echo
if [ "$PRESENT" = 0 ]; then
    echo "Nothing to remove here."
else
    SETTINGS="(not present)"
    if [ -e .env ] || [ -e config/models.local.yaml ]; then
        SETTINGS=".env, config/models.local.yaml"
    fi
    echo "Found in this folder:"
    printf '  Downloaded ranking model    %-19s %s\n' ".models/" "$(size_of .models)"
    printf '  Installed packages          %-19s %s\n' ".venv/" "$(size_of .venv)"
    printf '  Your profile and index      %-19s %s\n' "02_database/" "$(size_of 02_database)"
    printf '  Your applications and PDFs  %-19s %s\n' "04_applications/" "$(size_of 04_applications)"
    printf '  Copies of uploaded résumés  %-19s %s\n' ".gradio/" "$(size_of .gradio)"
    printf '  Your saved settings         %s\n' "$SETTINGS"

    # ------------------------------------------------------------ question 1

    echo
    cat <<'EOF'
1/2  Remove the downloaded ranking model and the installed packages (.models/ and .venv/)?
     This frees the space above. Your profile and applications are not touched.
EOF
    FAILED=0
    if yes_no n "     Double-clicking start.command later installs them again. [y/N]"; then
        rm -rf -- ./.models
        still_there ./.models
        rm -rf -- ./.venv
        still_there ./.venv
        if [ "$FAILED" = 0 ]; then
            echo "Removed .models/ and .venv/."
        fi
    else
        echo "Kept .models/ and .venv/."
    fi

    # ------------------------------------------------------------ question 2

    echo
    cat <<'EOF'
2/2  Also remove YOUR OWN DATA?
       02_database/       your profile and its search index
       04_applications/   every application, with its résumé and PDF
       .gradio/           copies of the résumé files you uploaded
       .env and config/models.local.yaml   your saved model settings, including your API key
EOF
    FAILED=0
    if yes_no n "     This cannot be undone. If you want to keep a PDF, answer no and copy it out first. [y/N]"; then
        rm -rf -- ./02_database
        still_there ./02_database
        rm -rf -- ./04_applications
        still_there ./04_applications
        rm -rf -- ./.gradio
        still_there ./.gradio
        rm -rf -- ./.env
        still_there ./.env
        rm -rf -- ./config/models.local.yaml
        still_there ./config/models.local.yaml
        if [ "$FAILED" = 0 ]; then
            echo "Removed your profile, your applications, the copies of your uploaded résumés and your saved settings."
        fi
    else
        echo "Kept your profile, your applications, the copies of your uploaded résumés and your saved settings."
    fi
fi

# ---------------------------------------------------------------- closing

echo
cat <<'EOF'
Done.

Not removed by this script:
  - The LoomCV folder itself: drag it to the Trash.
  - uv: see "Uninstall" in the README.
  - Your model server and its models: see "Uninstall" in the README.
EOF
