#!/bin/zsh
# Start from the project directory even when launched through Finder.
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print "Python environment missing. Follow the setup steps in README.md first."
  read "?Press Enter to close."
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py --server.headless=false
