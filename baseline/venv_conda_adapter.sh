#!/usr/bin/env bash
# Adapter for LingBot's conda-based evaluator when bootstrap.sh created venvs.
_lingbot_baseline_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda() {
  case "${1:-}" in
    activate)
      case "${2:-}" in
        lingbotvla) source "${_lingbot_baseline_dir}/../.venvs/lingbot/bin/activate" ;;
        RoboTwin) source "${_lingbot_baseline_dir}/../.venvs/robotwin/bin/activate" ;;
        *) echo "Unknown environment: ${2:-}" >&2; return 2 ;;
      esac
      ;;
    deactivate)
      if declare -F deactivate >/dev/null; then deactivate; fi
      ;;
    *) echo "Unsupported conda command: ${1:-}" >&2; return 2 ;;
  esac
}
