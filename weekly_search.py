"""
Búsqueda semanal automática de convocatorias para Fundación Bambalinas Co.
Llama a la API de Claude (con búsqueda web activada), arma el reporte,
y lo manda por correo. Pensado para correr una vez por semana vía GitHub Actions.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import anthropic

# ---------------------------------------------------------------------------
# 1. El prompt de búsqueda (el mismo que ya armamos, listo para reutilizar)
# ---------------------------------------------------------------------------
PROMPT = """
Quiero que hagas un barrido semanal de fondos y oportunidades para Fundación
Bambalinas Co. y para mi perfil personal (Samuel Iguarán). Dale un vistazo a
cada fuente y prioriza velocidad y cobertura amplia sobre verificación
exhaustiva de cada una — no necesitas confirmar cada dato en la fuente
oficial, solo dame lo que parezca más prometedor y yo verifico después las
que me interesen. Descarta cualquier cosa que sea préstamo, deuda o capital
reembolsable: solo quiero subvenciones no reembolsables (grants), becas o
premios.

PARTE 1 — Convocatorias para Bambalinas (organización)

Revisa estas fuentes y dime qué hay nuevo o próximo a cerrar esta semana.
Prioriza lo que encaje con: artesanas indígenas Wayuu, Sistema Artesanal,
economía comunitaria/rural, empleabilidad de jóvenes y migrantes, desarrollo
territorial en La Guajira, y organizaciones sociales/fortalecimiento
institucional.

Internacionales:
1. fundsforNGOs.org
2. Terra Viva Grants Directory (terravivagrants.org/funding-news)
3. DevelopmentAid.org
4. Wepropel — Oportunidades (wepropel.org/oportunidades)
5. Difusión con Causa (difusionconcausa.com/convocatorias)
6. Mercociudades — Oportunidades (sursurmercociudades.org/oportunidades)
7. Terraética — Mapa de donantes (terraetica.com/mapa-de-donantes)
8. FIMI / Fondo Ayni (fimi-iiwf.org/convocatorias)
9. IFAD — IPAF (ifad.org/ipaf)
10. AECID — convocatorias vigentes (aecid.es)
11. Fondation Botnar — funding opportunities (fondationbotnar.org/funding-opportunities)
12. Zendesk Tech for Good (techforgood.zendesk.com)
13. Thousand Currents — novedades/blog (thousandcurrents.org)
14. YouthOp (youthop.com)

Nacionales (Colombia):
15. APC Colombia — buscador de convocatorias (apccolombia.gov.co/buscador)
16. Artesanías de Colombia — convocatorias (artesaniasdecolombia.com.co)
17. Prosperidad Social — Economía Popular para el Cambio (prosperidadsocial.gov.co)
18. SENA — Fondo Emprender (fondoemprender.com)
19. Cámara de Comercio de La Guajira (camaraguajira.org)
20. Unidad para las Víctimas — línea Semillas (unidadvictimas.gov.co)
21. Innpulsa Colombia — convocatorias (innpulsacolombia.com)

Para cada hallazgo relevante dame: nombre, monto, fecha límite, link oficial,
y una línea de por qué encaja con Bambalinas.

PARTE 2 — Fellowships, becas y oportunidades personales (Samuel Iguarán)

Busca becas, fellowships y programas de liderazgo para mi perfil: Director
General de fundación social, background en diseño (Uniandes), enfoque en
innovación social, comunidades indígenas y desarrollo territorial en
Colombia. Incluye: fellowships de liderazgo (tipo YLAI, Ashoka, Draper
Richards Kaplan Fellow), becas de posgrado o programas ejecutivos cortos para
líderes sociales latinoamericanos, y premios individuales a
emprendedores/líderes sociales.

Dame máximo 5 opciones con: nombre, qué ofrece, fecha límite, link oficial,
y por qué encaja con mi perfil.

Formatea todo el resultado en HTML simple (usa <h2>, <h3>, <ul>, <li>, <a
href="...">), listo para pegar en el cuerpo de un correo. No uses markdown.
"""

# ---------------------------------------------------------------------------
# 2. Llamar a Claude con búsqueda web activada
# ---------------------------------------------------------------------------
def run_search() -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": PROMPT}],
    )

    html_parts = []
    for block in response.content:
        if block.type == "text":
            html_parts.append(block.text)

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# 3. Enviar el resultado por correo (usa Gmail con contraseña de aplicación)
# ---------------------------------------------------------------------------
def send_email(html_body: str) -> None:
    sender = os.environ["EMAIL_FROM"]
    password = os.environ["EMAIL_APP_PASSWORD"]
    recipient = os.environ["EMAIL_TO"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Barrido semanal de convocatorias — Bambalinas"
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())


if __name__ == "__main__":
    result_html = run_search()
    send_email(result_html)
    print("Listo — correo enviado.")
