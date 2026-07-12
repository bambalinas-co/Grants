"""
Barrido semanal de convocatorias — Fundación Bambalinas Co.

Arquitectura:
  1. Le pedimos a Claude datos ESTRUCTURADOS (JSON), no un correo ya escrito.
  2. Filtramos en Python lo que nunca debe pasar (préstamos/deuda) — un filtro
     de código no depende de que el modelo "se acuerde" cada semana.
  3. Calculamos un score compuesto por fórmula (no por intuición del modelo):
       score = 0.40 * fit_score          (encaje temático, lo estima el modelo)
             + 0.25 * urgency_score      (calculado en Python desde la fecha real)
             + 0.20 * amount_score       (escala logarítmica del monto)
             + 0.15 * geo_score          (Colombia > LatAm > global)
  4. Recordamos qué ya se reportó (state.json) para no repetir cada semana,
     salvo un recordatorio final si el plazo está por cerrar.
  5. Reintentos con backoff, logging, y un correo de error si algo falla —
     nunca un silencio total.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import smtplib
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("bambalinas-grants")

STATE_PATH = Path(__file__).parent / "state.json"
MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Palabras que descalifican una oportunidad sin importar qué tan bien "encaje"
# temáticamente. Esto es un filtro de código — no depende del prompt.
# ---------------------------------------------------------------------------
DISQUALIFYING_TERMS = [
    "loan", "debt instrument", "repayable", "reembolsable",
    "préstamo", "prestamo", "capital reembolsable", "line of credit",
    "equity investment", "debt financing",
]

COLOMBIA_TERMS = ["colombia", "colombian", "bogotá", "bogota", "la guajira", "wayuu"]
LATAM_TERMS = [
    "latin america", "latinoamérica", "latinoamerica", "latam",
    "america latina", "américa latina", "caribbean", "caribe",
    "andean", "andino",
]

PROMPT_TEMPLATE = """
Busca convocatorias, grants, becas y fellowships REALMENTE abiertos hoy
({today}) en las siguientes fuentes, para dos categorías distintas.

CATEGORÍA "bambalinas" — para Fundación Bambalinas Co. (organización):
Fuentes a revisar: fundsforNGOs.org, Terra Viva Grants Directory
(terravivagrants.org), DevelopmentAid.org, Wepropel (wepropel.org/oportunidades),
Difusión con Causa (difusionconcausa.com), Mercociudades
(sursurmercociudades.org/oportunidades), Terraética (terraetica.com/mapa-de-donantes),
FIMI/Fondo Ayni (fimi-iiwf.org), IFAD IPAF, AECID, Fondation Botnar
(fondationbotnar.org/funding-opportunities), Zendesk Tech for Good, APC Colombia
(apccolombia.gov.co/buscador), Artesanías de Colombia, Prosperidad Social,
SENA Fondo Emprender, Cámara de Comercio de La Guajira, Unidad para las Víctimas,
Innpulsa Colombia.
Prioriza: artesanas indígenas Wayuu, Sistema Artesanal, economía comunitaria/rural,
empleabilidad de jóvenes y migrantes, desarrollo territorial en La Guajira,
fortalecimiento organizacional.

CATEGORÍA "personal" — para Samuel Iguarán (Director General, background en
diseño, YLAI Fellow, liderazgo en innovación social y comunidades indígenas):
Fuentes a revisar: YouthOp (youthop.com), OpportunitiesCorners
(opportunitiescorners.com), fundsforNGOs Individuals
(fundsforindividuals.fundsforngos.org). Busca fellowships de liderazgo,
becas de posgrado o programas ejecutivos cortos, premios individuales.

REGLA DURA SOBRE FECHAS — esto es crítico, léelo con cuidado:
Antes de reportar cualquier fecha límite, verifica explícitamente en la
fuente oficial (no en un agregador ni de memoria) si esa fecha ya pasó
respecto a hoy ({today}). Nunca reportes una convocatoria cuya fecha límite
ya pasó.
Distingue estos tres casos con precisión:
  - Si la fuente CONFIRMA explícitamente que no hay fecha límite fija
    (programa permanente/continuo): deadline = "rolling", date_confidence = "verified".
  - Si encontraste una fecha específica en la fuente oficial: deadline =
    "YYYY-MM-DD", date_confidence = "verified".
  - Si NO pudiste confirmar la fecha con certeza (la fuente no la menciona
    claramente, o la información es de un agregador sin fecha verificable):
    deadline = "unknown", date_confidence = "unverified". NO asumas
    "rolling" solo porque no encontraste la fecha — eso es una suposición,
    no una verificación.

REGLA DURA: nunca incluyas préstamos, deuda, capital reembolsable, ni
inversión de equity. Solo subvenciones no reembolsables (grants), becas,
o premios.

Responde ÚNICAMENTE con un JSON válido (sin texto antes o después, sin
```json```), con esta forma exacta:

{{
  "opportunities": [
    {{
      "category": "bambalinas" o "personal",
      "name": "nombre de la convocatoria",
      "org": "organización que la ofrece",
      "amount_usd": <número entero en USD, o null si no aplica/no se sabe>,
      "deadline": "YYYY-MM-DD", "rolling", o "unknown",
      "date_confidence": "verified" o "unverified",
      "geography": "descripción breve del alcance geográfico",
      "type": "grant" | "fellowship" | "prize" | "other",
      "link": "URL oficial",
      "fit_score": <entero 0-10, qué tan bien encaja con el perfil>,
      "summary": "1-2 frases explicando de qué se trata y por qué encaja"
    }}
  ]
}}

Si una fuente no tiene nada relevante hoy, simplemente no incluyas nada de
ella — no inventes resultados para rellenar.
"""


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------
@dataclass
class Opportunity:
    category: str
    name: str
    org: str
    amount_usd: float | None
    deadline: str
    date_confidence: str
    geography: str
    type: str
    link: str
    fit_score: float
    summary: str
    composite_score: float = field(default=0.0)
    is_new: bool = field(default=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Opportunity":
        return cls(
            category=str(d.get("category", "bambalinas")),
            name=str(d.get("name", "Sin nombre")),
            org=str(d.get("org", "")),
            amount_usd=_safe_float(d.get("amount_usd")),
            deadline=str(d.get("deadline") or "unknown"),
            date_confidence=str(d.get("date_confidence") or "unverified"),
            geography=str(d.get("geography", "")),
            type=str(d.get("type", "grant")),
            link=str(d.get("link", "")),
            fit_score=_safe_float(d.get("fit_score")) or 0.0,
            summary=str(d.get("summary", "")),
        )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1. Llamada a la API con reintentos y backoff exponencial
# ---------------------------------------------------------------------------
def call_claude_with_retries(client: anthropic.Anthropic, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("Llamando a la API de Claude (intento %d/%d)", attempt, MAX_RETRIES)
            response = client.messages.create(
                model=MODEL,
                max_tokens=8000,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks)
        except Exception as exc:  # noqa: BLE001 — queremos capturar cualquier fallo de red/API
            last_error = exc
            wait = 2 ** attempt
            log.warning("Fallo en intento %d: %s — reintentando en %ds", attempt, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"La API de Claude falló tras {MAX_RETRIES} intentos") from last_error


def parse_opportunities(raw_text: str) -> list[Opportunity]:
    """Extrae el JSON de la respuesta, tolerando que venga envuelto en texto
    o en fences de markdown pese a habérselo pedido explícitamente."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Si aún así hay texto antes/después del objeto JSON, recorta al primer
    # '{' y al último '}' que hagan match razonable.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

    data = json.loads(cleaned)
    raw_opportunities = data.get("opportunities", [])
    return [Opportunity.from_dict(o) for o in raw_opportunities]


# ---------------------------------------------------------------------------
# 2. Filtro duro — descalifica préstamos/deuda sin importar el fit_score
# ---------------------------------------------------------------------------
def is_disqualified(opp: Opportunity) -> bool:
    haystack = f"{opp.name} {opp.type} {opp.summary}".lower()
    if any(term in haystack for term in DISQUALIFYING_TERMS):
        return True
    return is_expired(opp.deadline)


def is_expired(deadline_str: str, today: date | None = None) -> bool:
    """Una fecha ya pasada descalifica sin importar qué tan bien encaje —
    a diferencia de la urgencia (que solo resta puntos), esto elimina."""
    if today is None:
        today = date.today()
    if deadline_str == "rolling" or not deadline_str:
        return False
    try:
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date()
    except ValueError:
        return False
    return deadline < today


# ---------------------------------------------------------------------------
# 3. Cálculo del score compuesto
# ---------------------------------------------------------------------------
def urgency_score(deadline_str: str, today: date) -> float:
    """Entre más cerca el plazo (sin haber pasado), más urgente — pero algo
    que cierra en menos de 2 días puntúa un poco menos porque probablemente
    ya no da tiempo de armar una buena postulación.
    'rolling' (confirmado sin fecha fija) puntúa neutral. 'unknown' (no se
    pudo verificar) puntúa más bajo — la incertidumbre no debe premiarse."""
    if deadline_str == "rolling":
        return 5.0
    if deadline_str == "unknown" or not deadline_str:
        return 3.0
    try:
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date()
    except ValueError:
        return 3.0

    days_left = (deadline - today).days
    if days_left < 0:
        return 0.0  # ya cerró
    if days_left <= 2:
        return 6.0
    if days_left <= 14:
        return 10.0
    if days_left <= 30:
        return 8.0
    if days_left <= 60:
        return 6.0
    if days_left <= 120:
        return 4.0
    return 2.0


def amount_score(amount_usd: float | None) -> float:
    """Escala logarítmica: la diferencia entre $1,000 y $10,000 importa más,
    perceptualmente, que la diferencia entre $500,000 y $509,000."""
    if amount_usd is None or amount_usd <= 0:
        return 3.0  # desconocido: no penaliza demasiado, pero no suma
    capped = min(amount_usd, 2_000_000)
    return min(10.0, (math.log10(capped) / math.log10(2_000_000)) * 10)


def geo_score(geography: str) -> float:
    text = geography.lower()
    if any(term in text for term in COLOMBIA_TERMS):
        return 10.0
    if any(term in text for term in LATAM_TERMS):
        return 7.0
    if "global" in text or "worldwide" in text or "world" in text:
        return 4.0
    return 3.0


def compute_composite(opp: Opportunity, today: date) -> float:
    fit = max(0.0, min(10.0, opp.fit_score))
    urgency = urgency_score(opp.deadline, today)
    amount = amount_score(opp.amount_usd)
    geo = geo_score(opp.geography)
    raw = 0.40 * fit + 0.25 * urgency + 0.20 * amount + 0.15 * geo

    # Penalización por baja confianza: una fecha no verificada no debe
    # competir en igualdad de condiciones con una confirmada en la fuente.
    confidence_multiplier = 1.0 if opp.date_confidence == "verified" else 0.85
    return round(raw * confidence_multiplier, 2)


# ---------------------------------------------------------------------------
# 4. Memoria entre semanas (deduplicación)
# ---------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            log.warning("state.json corrupto — arrancando de cero")
    return {"seen": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def apply_dedup(opportunities: list[Opportunity], state: dict[str, Any], today: date) -> list[Opportunity]:
    seen: dict[str, Any] = state.setdefault("seen", {})
    result: list[Opportunity] = []

    for opp in opportunities:
        key = opp.link or f"{opp.name}::{opp.org}"
        previously_seen = key in seen

        if previously_seen:
            opp.is_new = False
            deadline_soon = urgency_score(opp.deadline, today) >= 8.0
            if not deadline_soon:
                continue  # ya se reportó y no es urgente todavía: se omite
        else:
            opp.is_new = True
            seen[key] = {"first_seen": today.isoformat(), "name": opp.name}

        result.append(opp)

    return result


# ---------------------------------------------------------------------------
# 5. Construcción del correo (HTML generado en Python, no confiado al modelo)
# ---------------------------------------------------------------------------
def score_bar(score: float) -> str:
    filled = round(score)
    return "&#9632;" * filled + "&#9633;" * (10 - filled)  # ■■■□□□□□□□


def render_opportunity(opp: Opportunity) -> str:
    status_badge = '<span style="background:#2f6f4f;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;">NUEVA</span>' if opp.is_new else '<span style="background:#b45309;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;">CIERRA PRONTO</span>'
    warning_badge = ""
    if opp.date_confidence != "verified":
        warning_badge = '&nbsp;<span style="background:#dc2626;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;">&#9888; FECHA SIN CONFIRMAR</span>'
    amount_text = f"${opp.amount_usd:,.0f} USD" if opp.amount_usd else "Monto no especificado"
    return f"""
    <tr>
      <td style="padding:14px 0;border-bottom:1px solid #e5e5e5;">
        <div style="font-size:15px;font-weight:600;color:#1a1a1a;">
          <a href="{opp.link}" style="color:#1a1a1a;text-decoration:none;">{opp.name}</a>
          &nbsp;{status_badge}{warning_badge}
        </div>
        <div style="font-size:13px;color:#555;margin-top:2px;">
          {opp.org} &middot; {amount_text} &middot; Cierra: {opp.deadline}
        </div>
        <div style="font-size:13px;color:#333;margin-top:6px;">{opp.summary}</div>
        <div style="font-size:12px;color:#888;margin-top:6px;font-family:monospace;">
          Score {opp.composite_score:.1f}/10 &nbsp;{score_bar(opp.composite_score)}
        </div>
      </td>
    </tr>
    """


def render_section(title: str, items: list[Opportunity]) -> str:
    if not items:
        return f"<h2 style='font-family:sans-serif;color:#1a1a1a;'>{title}</h2><p style='font-family:sans-serif;color:#888;'>Nada nuevo ni urgente esta semana.</p>"
    rows = "\n".join(render_opportunity(o) for o in items)
    return f"""
    <h2 style="font-family:sans-serif;color:#1a1a1a;border-bottom:2px solid #1a1a1a;padding-bottom:6px;">{title}</h2>
    <table style="width:100%;border-collapse:collapse;font-family:sans-serif;">{rows}</table>
    """


def build_email_html(bambalinas: list[Opportunity], personal: list[Opportunity], today: date) -> str:
    return f"""
    <div style="max-width:640px;margin:0 auto;font-family:sans-serif;">
      <p style="color:#888;font-size:12px;">Barrido semanal &middot; {today.isoformat()}</p>
      {render_section("Fundación Bambalinas Co.", bambalinas)}
      {render_section("Oportunidades personales", personal)}
      <p style="color:#aaa;font-size:11px;margin-top:24px;">
        Generado automáticamente. Los préstamos y capital reembolsable se filtran
        siempre, sin excepción. El score combina encaje temático, urgencia,
        tamaño del monto y alcance geográfico.
      </p>
    </div>
    """


def build_error_email_html(error: Exception) -> str:
    return f"""
    <div style="font-family:sans-serif;">
      <h2 style="color:#b91c1c;">El barrido semanal falló</h2>
      <p>No se pudo completar la búsqueda esta semana. Detalle técnico:</p>
      <pre style="background:#f5f5f5;padding:12px;border-radius:6px;white-space:pre-wrap;">{error}</pre>
      <p>Revisa los logs en la pestaña "Actions" del repositorio para más detalle.</p>
    </div>
    """


# ---------------------------------------------------------------------------
# 6. Envío de correo
# ---------------------------------------------------------------------------
def send_email(subject: str, html_body: str) -> None:
    sender = os.environ["EMAIL_FROM"]
    password = os.environ["EMAIL_APP_PASSWORD"]
    recipient = os.environ["EMAIL_TO"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
    log.info("Correo enviado a %s", recipient)


# ---------------------------------------------------------------------------
# Orquestación principal
# ---------------------------------------------------------------------------
def main() -> None:
    today = date.today()
    required_env = ["ANTHROPIC_API_KEY", "EMAIL_FROM", "EMAIL_APP_PASSWORD", "EMAIL_TO"]
    missing = [v for v in required_env if v not in os.environ]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        prompt = PROMPT_TEMPLATE.format(today=today.isoformat())
        raw = call_claude_with_retries(client, prompt)
        opportunities = parse_opportunities(raw)
        log.info("Se recibieron %d oportunidades en bruto", len(opportunities))

        opportunities = [o for o in opportunities if not is_disqualified(o)]
        log.info("Quedan %d tras filtrar préstamos/deuda", len(opportunities))

        for opp in opportunities:
            opp.composite_score = compute_composite(opp, today)

        state = load_state()
        opportunities = apply_dedup(opportunities, state, today)
        opportunities.sort(key=lambda o: o.composite_score, reverse=True)

        bambalinas = [o for o in opportunities if o.category == "bambalinas"][:15]
        personal = [o for o in opportunities if o.category == "personal"][:5]

        html = build_email_html(bambalinas, personal, today)
        send_email(f"Barrido semanal de convocatorias — {today.isoformat()}", html)

        save_state(state)
        log.info("Listo. state.json actualizado con %d links vistos.", len(state["seen"]))

    except Exception as exc:  # noqa: BLE001
        log.exception("El barrido falló")
        try:
            send_email(f"[ERROR] Barrido semanal — {today.isoformat()}", build_error_email_html(exc))
        except Exception:  # noqa: BLE001
            log.exception("Ni siquiera se pudo enviar el correo de error")
        raise


if __name__ == "__main__":
    main()
