from __future__ import annotations

import subprocess


class ChromeTabsError(RuntimeError):
    pass


def capture_chrome_urls(all_windows: bool = False) -> list[str]:
    raw_output = _run_osascript(_script_lines(all_windows=all_windows))
    urls = _parse_osascript_output(raw_output)

    if not urls:
        raise ChromeTabsError("Google Chrome has no valid http(s) tabs in the selected window scope.")

    return urls


def _script_lines(all_windows: bool) -> list[str]:
    if all_windows:
        tab_loop = [
            'repeat with chrome_window in windows',
            'repeat with chrome_tab in tabs of chrome_window',
            'set end of url_list to (URL of chrome_tab)',
            'end repeat',
            'end repeat',
        ]
    else:
        tab_loop = [
            'repeat with chrome_tab in tabs of front window',
            'set end of url_list to (URL of chrome_tab)',
            'end repeat',
        ]

    return [
        'if application "Google Chrome" is not running then error "GOOGLE_CHROME_NOT_RUNNING" number 1001',
        'tell application "Google Chrome"',
        'if (count of windows) is 0 then error "GOOGLE_CHROME_NO_WINDOWS" number 1002',
        'set url_list to {}',
        *tab_loop,
        'end tell',
        "set AppleScript's text item delimiters to linefeed",
        "return url_list as text",
    ]


def _run_osascript(lines: list[str]) -> str:
    command = ["osascript"]
    for line in lines:
        command.extend(["-e", line])

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if completed.returncode == 0:
        return stdout

    combined = "\n".join(part for part in (stdout, stderr) if part)
    if "GOOGLE_CHROME_NOT_RUNNING" in combined:
        raise ChromeTabsError("Google Chrome is not running.")
    if "GOOGLE_CHROME_NO_WINDOWS" in combined:
        raise ChromeTabsError("Google Chrome has no open windows.")
    if any(token in combined for token in ("Not authorized to send Apple events", "(-1743)", "1743")):
        raise ChromeTabsError(
            "Automation permission denied. Allow your terminal app to control Google Chrome in System Settings."
        )

    raise ChromeTabsError(f"Failed to read Chrome tabs via osascript: {combined or 'unknown error'}")


def _parse_osascript_output(raw_output: str) -> list[str]:
    urls: list[str] = []
    for line in raw_output.splitlines():
        url = line.strip()
        if _is_supported_url(url):
            urls.append(url)
    return urls


def _is_supported_url(url: str) -> bool:
    return url.startswith("https://") or url.startswith("http://")
