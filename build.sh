#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# build.sh - Build the qucomp dissertation PDF.
#
# Originally written 2026 to replace the 2004 makefile (which targeted the
# now-unavailable LaTeX -> dvips -> dvipdf toolchain). This script uses
# pdflatex + bibtex from any modern TeX Live distribution and runs on both
# macOS and Linux.
#
# Usage:
#   ./build.sh              # build qucomp.pdf
#   ./build.sh clean        # remove build artefacts (.aux, .log, etc.)
#   ./build.sh distclean    # also remove the generated PDF and DVI
#
# Requirements (any modern TeX distribution):
#   pdflatex, bibtex
#
# Install hints:
#   macOS:   brew install --cask basictex
#            sudo tlmgr update --self
#            sudo tlmgr install latexmk geometry txfonts setspace fancyhdr \
#                                tabularx lscape rotating listings xcolor   \
#                                hyperref multirow
#
#   Debian / Ubuntu:
#            sudo apt-get install texlive-latex-recommended    \
#                                  texlive-latex-extra texlive-fonts-extra \
#                                  texlive-bibtex-extra biber
#
#   Fedora:  sudo dnf install texlive-scheme-medium
#
# -----------------------------------------------------------------------------
set -euo pipefail

# --- paths -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/source code"
MAIN="qucomp"

# --- subcommands -------------------------------------------------------------
ACTION="${1:-build}"

# --- artefacts cleanup -------------------------------------------------------
ARTEFACTS=(aux log toc out lof lot bbl blg idx ilg ind fls fdb_latexmk synctex.gz)

clean_artefacts() {
    local pattern_dir="$1"
    (
        cd "$pattern_dir" || return 0
        # nullglob: unmatched globs expand to nothing instead of literal "*.foo".
        # Done in a subshell so the shopt change doesn't leak.
        shopt -s nullglob
        for ext in "${ARTEFACTS[@]}"; do
            # Match every .ext file in the directory (no recursion).
            # Some old builds wrote secondary aux files for every \include'd file
            # (e.g. introduction.aux), so this catches those too.
            for f in *."$ext"; do
                rm -f "$f"
            done
        done
    )
    return 0
}

# --- handle clean / distclean ------------------------------------------------
case "$ACTION" in
    clean)
        echo "==> cleaning build artefacts in $SRC_DIR"
        clean_artefacts "$SRC_DIR"
        exit 0
        ;;
    distclean)
        echo "==> cleaning build artefacts and outputs"
        clean_artefacts "$SRC_DIR"
        rm -f "${SRC_DIR}/${MAIN}.pdf" "${SRC_DIR}/${MAIN}.dvi"
        rm -f "${SCRIPT_DIR}/${MAIN}.pdf"
        exit 0
        ;;
    build|"")
        # fall through to build below
        ;;
    *)
        echo "Usage: $0 [build|clean|distclean]" >&2
        exit 1
        ;;
esac

# --- preflight: ensure pdflatex and bibtex are available ---------------------
require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        cat <<EOF >&2

ERROR: '$cmd' is not in PATH.

This script needs a working TeX distribution. Suggested installs:

  macOS:
      brew install --cask basictex
      eval "\$(/usr/libexec/path_helper)"
      sudo tlmgr update --self
      sudo tlmgr install latexmk geometry txfonts setspace fancyhdr \\
                         tabularx lscape rotating listings xcolor hyperref

  Debian / Ubuntu:
      sudo apt-get install texlive-latex-recommended texlive-latex-extra \\
                            texlive-fonts-extra texlive-bibtex-extra

  Fedora:
      sudo dnf install texlive-scheme-medium

After installing on macOS, you may need to open a new shell or run:
      eval "\$(/usr/libexec/path_helper)"

EOF
        exit 127
    fi
}

# macOS: make sure /Library/TeX/texbin (BasicTeX / MacTeX) is on PATH even
# inside non-login shells that haven't sourced /etc/paths.d.
if [[ "$(uname -s)" == "Darwin" && -d "/Library/TeX/texbin" ]]; then
    case ":$PATH:" in
        *":/Library/TeX/texbin:"*) ;;
        *) export PATH="/Library/TeX/texbin:$PATH" ;;
    esac
fi

require_command pdflatex
require_command bibtex

# --- run the build -----------------------------------------------------------
cd "$SRC_DIR"

echo "==> pdflatex (pass 1)"
pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex" >/dev/null
echo "==> bibtex"
bibtex "${MAIN}" || {
    echo "WARNING: bibtex returned non-zero; check ${MAIN}.blg for details." >&2
}
echo "==> pdflatex (pass 2)"
pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex" >/dev/null
echo "==> pdflatex (pass 3, resolve refs)"
pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex" >/dev/null

# --- copy PDF to repo root and clean up --------------------------------------
cp "${MAIN}.pdf" "${SCRIPT_DIR}/${MAIN}.pdf"

echo "==> cleaning intermediate artefacts"
clean_artefacts "$SRC_DIR"

echo ""
echo "OK: ${SCRIPT_DIR}/${MAIN}.pdf  ($(wc -c < "${SCRIPT_DIR}/${MAIN}.pdf") bytes)"
