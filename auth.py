"""
Login for the web UI: username and password, then a one-time code sent by email through Resend.

Turned on when FRAZILE_AUTH_USERNAME is set (the Docker app behind ai.frazile.si); without it the
app stays open as before (run-app.bat on this PC).

Environment:
    FRAZILE_AUTH_USERNAME    login name
    FRAZILE_AUTH_PASSWORD    login password
    FRAZILE_AUTH_EMAIL       where the one-time code is sent
    RESEND_API_KEY           Resend API key (re_...)
    FRAZILE_OTP_FROM         sender, default "Frazile AI <onboarding@resend.dev>"; use an address on a
                             domain verified in Resend (the default only delivers to the Resend account owner)
    FRAZILE_SESSION_HOURS    how long a login lasts, default 12

Flow: /login (username + password) -> code emailed -> /login/code -> signed session cookie.
Every other path, the Gradio API included, needs that cookie. /logout ends the session.
"""
import hashlib
import hmac
import html
import os
import secrets
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import applog
from ui_theme import ACCENT, BG, BORDER, GRADIENT, MUTED, SURFACE, TEXT

SESSION_COOKIE, PENDING_COOKIE = "fz_session", "fz_pending"
CODE_TTL, CODE_ATTEMPTS = 10 * 60, 5                  # a code lasts 10 minutes and allows 5 guesses
LOCK_AFTER, LOCK_SECONDS = 5, 15 * 60                 # 5 wrong passwords from one address -> 15 minute lock
OPEN_PATHS = ("/login", "/logout", "/favicon.ico")
SECRET_FILE = Path(__file__).parent / "models" / "auth_secret.key"

_pending: dict[str, dict] = {}                        # pending token -> {code hash, expiry, attempts}
_failures: dict[str, list[float]] = {}                # client address -> times of wrong passwords


def enabled() -> bool:
    return bool(os.environ.get("FRAZILE_AUTH_USERNAME"))


def check_config() -> None:
    missing = [k for k in ("FRAZILE_AUTH_PASSWORD", "FRAZILE_AUTH_EMAIL", "RESEND_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"FRAZILE_AUTH_USERNAME is set, so login is on, but these are missing: {', '.join(missing)}")


def _secret() -> bytes:
    # kept in models/ (a volume in Docker) so logins survive a restart
    if not SECRET_FILE.exists():
        SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        SECRET_FILE.write_text(secrets.token_hex(32))
    return bytes.fromhex(SECRET_FILE.read_text().strip())


def _sign(value: str) -> str:
    return hmac.new(_secret(), value.encode(), hashlib.sha256).hexdigest()


def _hash_code(token: str, code: str) -> str:
    return _sign(f"code|{token}|{code}")


def session_user(request: Request) -> str | None:
    raw = request.cookies.get(SESSION_COOKIE, "")
    try:
        user, expires, sig = raw.rsplit("|", 2)
        if hmac.compare_digest(sig, _sign(f"{user}|{expires}")) and float(expires) > time.time():
            return user
    except ValueError:
        pass
    return None


def _client(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _secure(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


def _locked(ip: str) -> bool:
    recent = [t for t in _failures.get(ip, []) if t > time.time() - LOCK_SECONDS]
    _failures[ip] = recent
    return len(recent) >= LOCK_AFTER


def _send_code(code: str) -> None:
    sender = os.environ.get("FRAZILE_OTP_FROM") or "Frazile AI <onboarding@resend.dev>"
    body = (f'<div style="font-family:sans-serif"><p>Your Frazile AI sign-in code:</p>'
            f'<p style="font-size:32px;font-weight:700;letter-spacing:6px">{code}</p>'
            f'<p style="color:#666">It expires in {CODE_TTL // 60} minutes. If you did not try to sign in, '
            f'change your password.</p></div>')
    r = httpx.post("https://api.resend.com/emails", timeout=15,
                   headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}"},
                   json={"from": sender, "to": [os.environ["FRAZILE_AUTH_EMAIL"]],
                         "subject": f"Frazile AI sign-in code: {code}", "html": body})
    if r.status_code >= 300:
        raise RuntimeError(f"Resend returned {r.status_code}: {r.text[:300]}")


def _masked_email() -> str:
    name, _, domain = os.environ.get("FRAZILE_AUTH_EMAIL", "").partition("@")
    return f"{name[:2]}{'•' * max(len(name) - 2, 1)}@{domain}"


def _page(title: str, body: str, error: str = "") -> HTMLResponse:
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Frazile AI · {title}</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;min-height:100vh;display:grid;place-items:center;background:{BG};
color:{TEXT};font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:16px}}
.card{{width:100%;max-width:380px;background:{SURFACE};border:1px solid {BORDER};border-radius:14px;padding:28px}}
.brand{{font-weight:800;font-size:22px;background:{GRADIENT};-webkit-background-clip:text;background-clip:text;color:transparent}}
h1{{font-size:16px;font-weight:600;margin:4px 0 20px;color:{MUTED}}}
label{{display:block;font-size:13px;color:{MUTED};margin:14px 0 6px}}
input{{width:100%;padding:11px 12px;border-radius:9px;border:1px solid {BORDER};background:{BG};color:{TEXT};font-size:15px}}
input:focus{{outline:2px solid {ACCENT};outline-offset:1px}}
input.code{{font-size:26px;letter-spacing:10px;text-align:center}}
button{{width:100%;margin-top:22px;padding:12px;border:0;border-radius:9px;background:{GRADIENT};color:#1A0A12;
font-weight:700;font-size:15px;cursor:pointer}}
.err{{color:#f87171;font-size:14px;margin:12px 0 0}} .note{{color:{MUTED};font-size:13px;margin-top:16px}}
a{{color:{ACCENT}}}
</style></head><body><main class="card"><div class="brand">Frazile AI</div><h1>{title}</h1>{body}{err}</main></body></html>""")


def _login_form(error: str = "") -> HTMLResponse:
    return _page("Sign in", """<form method="post" action="/login">
<label for="u">Username</label><input id="u" name="username" autocomplete="username" required autofocus>
<label for="p">Password</label><input id="p" name="password" type="password" autocomplete="current-password" required>
<button type="submit">Continue</button></form>""", error)


def _code_form(error: str = "") -> HTMLResponse:
    return _page("Enter the code", f"""<form method="post" action="/login/code">
<p class="note" style="margin-top:0">We emailed a 6-digit code to {html.escape(_masked_email())}.</p>
<label for="c">Code</label><input id="c" class="code" name="code" inputmode="numeric" pattern="[0-9]{{6}}"
maxlength="6" autocomplete="one-time-code" required autofocus>
<button type="submit">Sign in</button></form>
<p class="note"><a href="/login">Start again</a></p>""", error)


def install(app: FastAPI) -> None:
    """Add the login routes and the middleware that guards everything else."""

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        if path in OPEN_PATHS or path.startswith("/login/") or session_user(request):
            return await call_next(request)
        if request.method == "GET" and (path == "/" or "text/html" in request.headers.get("accept", "")):
            return RedirectResponse("/login", status_code=303)
        return JSONResponse({"detail": "Sign in at /login"}, status_code=401)

    @app.get("/login")
    async def login_page(request: Request):
        if session_user(request):
            return RedirectResponse("/", status_code=303)
        return _login_form()

    @app.post("/login")
    async def login_submit(request: Request, username: str = Form(""), password: str = Form("")):
        ip = _client(request)
        if _locked(ip):
            return _login_form("Too many wrong attempts. Try again in 15 minutes.")
        ok_user = hmac.compare_digest(username.encode(), os.environ["FRAZILE_AUTH_USERNAME"].encode())
        ok_pass = hmac.compare_digest(password.encode(), os.environ["FRAZILE_AUTH_PASSWORD"].encode())
        if not (ok_user and ok_pass):
            _failures.setdefault(ip, []).append(time.time())
            applog.log(f"Login: wrong username or password from {ip}")
            return _login_form("Wrong username or password.")
        _failures.pop(ip, None)
        token, code = secrets.token_urlsafe(24), f"{secrets.randbelow(10**6):06d}"
        try:
            _send_code(code)
        except Exception as e:  # noqa: BLE001 - shown to the user, details in the log
            applog.log(f"Login: could not send the code: {e}")
            return _login_form("Could not send the email code. Check the Resend settings.")
        now = time.time()
        for t in [t for t, p in _pending.items() if p["expires"] < now]:
            del _pending[t]
        _pending[token] = {"hash": _hash_code(token, code), "expires": now + CODE_TTL, "attempts": 0}
        applog.log(f"Login: password accepted from {ip}, code emailed")
        resp = RedirectResponse("/login/code", status_code=303)
        resp.set_cookie(PENDING_COOKIE, token, max_age=CODE_TTL, httponly=True, samesite="lax", secure=_secure(request))
        return resp

    @app.get("/login/code")
    async def code_page(request: Request):
        if request.cookies.get(PENDING_COOKIE) not in _pending:
            return RedirectResponse("/login", status_code=303)
        return _code_form()

    @app.post("/login/code")
    async def code_submit(request: Request, code: str = Form("")):
        token = request.cookies.get(PENDING_COOKIE, "")
        pending = _pending.get(token)
        if not pending or pending["expires"] < time.time():
            _pending.pop(token, None)
            return _login_form("The code expired. Sign in again.")
        if not hmac.compare_digest(pending["hash"], _hash_code(token, code.strip())):
            pending["attempts"] += 1
            if pending["attempts"] >= CODE_ATTEMPTS:
                del _pending[token]
                return _login_form("Too many wrong codes. Sign in again.")
            return _code_form("Wrong code.")
        del _pending[token]
        user = os.environ["FRAZILE_AUTH_USERNAME"]
        expires = str(int(time.time() + float(os.environ.get("FRAZILE_SESSION_HOURS") or 12) * 3600))
        applog.log(f"Login: {user} signed in from {_client(request)}")
        resp = RedirectResponse("/", status_code=303)
        resp.delete_cookie(PENDING_COOKIE)
        resp.set_cookie(SESSION_COOKIE, f"{user}|{expires}|{_sign(f'{user}|{expires}')}",
                        max_age=int(float(expires) - time.time()), httponly=True, samesite="lax", secure=_secure(request))
        return resp

    @app.get("/logout")
    async def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(SESSION_COOKIE)
        return resp
