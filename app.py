# =============================================================================
#  Satellite Orbit Simulator – 2D · Keplersche Gesetze · Python
# =============================================================================
#
#  Start im Terminal (im Projektordner, mit aktiver virtueller Umgebung):
#
#      streamlit run app.py
#
# -----------------------------------------------------------------------------
#  WAS MACHT DAS PROGRAMM?
# -----------------------------------------------------------------------------
#  Das Programm simuliert einen Satelliten, der die Erde umkreist. Man stellt
#  über Schieberegler Höhe, Geschwindigkeit, Richtung der Geschwindigkeit und
#  Startposition ein. Das Programm berechnet daraus die Bahn und zeichnet sie.
#
#  Wichtig: Es gibt KEINE schrittweise Simulation der Kräfte (kein "Zeitschritt
#  für Zeitschritt"). Stattdessen nutzen wir, dass die Bewegung im Zweikörper-
#  problem (Erde + Satellit) exakt bekannt ist: Der Satellit bewegt sich auf
#  einer Kegelschnittbahn (hier: Ellipse). Es reicht also,
#
#     1. aus Startort und Startgeschwindigkeit die Form der Ellipse zu
#        berechnen (Funktion compute_orbit) und
#     2. für jeden gewünschten Zeitpunkt die Position auf dieser Ellipse
#        auszurechnen (Funktion state_at, nutzt die Kepler-Gleichung).
#
# -----------------------------------------------------------------------------
#  AUFBAU DER DATEI
# -----------------------------------------------------------------------------
#   ABSCHNITT 1: Imports
#   ABSCHNITT 2: Physikalische Konstanten und Einstellungen
#   ABSCHNITT 3: Hilfsfunktionen für die Zahlenformatierung
#   ABSCHNITT 4: Bahnmechanik (die eigentliche Physik)
#   ABSCHNITT 5: Darstellung mit Plotly (die Grafik)
#   ABSCHNITT 6: Callback-Funktionen für die Knöpfe
#   ABSCHNITT 7: Benutzeroberfläche mit Streamlit (der Hauptablauf)
#
# -----------------------------------------------------------------------------
#  KOORDINATENSYSTEM UND EINHEITEN
# -----------------------------------------------------------------------------
#  - Die Erde liegt immer im Koordinatenursprung (0, 0).
#  - x-Achse zeigt nach rechts, y-Achse nach oben.
#  - Winkel werden gegen den Uhrzeigersinn gemessen, 0° = Richtung +x.
#  - Alle Entfernungen sind in Kilometern (km), alle Zeiten in Sekunden (s),
#    Geschwindigkeiten in km/s. Winkel werden intern im Bogenmaß (rad) benutzt,
#    auf der Oberfläche aber in Grad (°) angezeigt.
# =============================================================================


# =============================================================================
# ABSCHNITT 1: IMPORTS
# =============================================================================
import math   # sin, cos, sqrt, atan2 usw. für einzelne Zahlen
import time   # nur für die kurze Pause zwischen zwei Animationsbildern

import numpy as np                  # Vektoren und Arrays (Orts- und Geschwindigkeitsvektoren,
                                    # Punkte der Ellipse)
import plotly.graph_objects as go   # zeichnet die interaktive Grafik
import streamlit as st              # baut die Benutzeroberfläche (Regler, Knöpfe, Spalten)


# =============================================================================
# ABSCHNITT 2: PHYSIKALISCHE KONSTANTEN UND EINSTELLUNGEN
# =============================================================================

# Mittlerer Erdradius in km.
EARTH_RADIUS = 6371.0

# Gravitationsparameter der Erde: mu = G * M_Erde (in km^3/s^2).
# Er fasst die Gravitationskonstante G und die Erdmasse M zusammen. Die
# Beschleunigung eines Satelliten im Abstand r vom Erdmittelpunkt ist
#     a = mu / r^2
# Die Masse des Satelliten kommt darin NICHT vor – deshalb hängt die Bahn nicht
# von der Satellitenmasse ab.
MU = 398600.4418

# Fallbeschleunigung an der Erdoberfläche in m/s^2. Wird nur gebraucht, um
# anzuzeigen, wie viel Prozent davon in der gewählten Höhe noch übrig sind.
G0 = 9.80665

# Der Zeit-Regler geht von 0 bis T_MAX. Die Zeit wird in "Umlaufzeiten" T
# angegeben (1.0 = ein voller Umlauf), weil dadurch der Regler unabhängig von
# der Bahn immer denselben Wertebereich hat.
T_MAX = 2.0

# Animationsschritt pro Bild (in Bruchteilen einer Umlaufzeit).
# Beispiel "Normal": 0.010 -> 100 Bilder pro Umlauf.
# Die Schlüssel (links) sieht der Benutzer, die Werte (rechts) braucht die Rechnung.
ANIM_STEPS = {
    "Langsam": 0.004,
    "Normal": 0.010,
    "Schnell": 0.030,
}


# =============================================================================
# ABSCHNITT 3: HILFSFUNKTIONEN FÜR DIE ZAHLENFORMATIERUNG
# =============================================================================
def fmt(x, digits=2):
    """
    Formatiert eine Zahl im deutschen Format: Punkt als Tausendertrennzeichen,
    Komma als Dezimaltrenner.

    Beispiel: fmt(1234567.891, 2)  ->  '1.234.567,89'

    Python formatiert Zahlen im amerikanischen Format ('1,234,567.89').
    Würden wir einfach "," durch "." und "." durch "," ersetzen, würde die
    zweite Ersetzung das Ergebnis der ersten wieder überschreiben. Deshalb
    wird in drei Schritten getauscht, mit dem Zeichen '§' als Zwischenspeicher.
    """
    s = f"{x:,.{digits}f}"                 # '1,234,567.89'  (amerikanisch)
    s = s.replace(",", "§")                # '1§234§567.89'  (Tausender merken)
    s = s.replace(".", ",")                # '1§234§567,89'  (Dezimalkomma)
    return s.replace("§", ".")             # '1.234.567,89'  (Tausenderpunkt)


def fmt_sci(x, digits=3):
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 9,870e-05."""
    return f"{x:.{digits}e}".replace(".", ",")


# =============================================================================
# ABSCHNITT 4: BAHNMECHANIK (DIE EIGENTLICHE PHYSIK)
# =============================================================================
def wrap_pi(x):
    """
    Bringt einen beliebigen Winkel (in rad) in den Bereich [-pi, +pi].

    Warum? Winkel wiederholen sich alle 2*pi (z. B. sind 7 rad und 7 - 2*pi
    rad dieselbe Richtung). Für die Rechnung ist ein eindeutiger Bereich
    praktisch.

    Wie? Wir verschieben den Winkel um +pi, nehmen den Rest bei der Division
    durch 2*pi (liefert in Python immer einen Wert zwischen 0 und 2*pi) und
    verschieben um -pi zurück.
    """
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def solve_kepler(M, e, tol=1e-12, max_iter=100):
    """
    Löst die Kepler-Gleichung  M = E - e * sin(E)  nach E auf.

    Bedeutung der Größen:
      M = mittlere Anomalie     (wächst gleichmäßig mit der Zeit)
      E = exzentrische Anomalie (Hilfswinkel, aus dem man den Ort berechnet)
      e = Exzentrizität der Bahn (0 = Kreis, nahe 1 = sehr lange Ellipse)

    Warum numerisch? Die Gleichung lässt sich nicht nach E umstellen, weil E
    einmal "frei" und einmal im Sinus vorkommt. Man löst sie deshalb mit dem
    NEWTON-VERFAHREN: Gesucht ist die Nullstelle der Funktion

        f(E) = E - e*sin(E) - M            (soll 0 werden)

    Die Ableitung ist f'(E) = 1 - e*cos(E). Das Newton-Verfahren verbessert
    eine Schätzung E immer wieder nach der Vorschrift

        E_neu = E_alt - f(E_alt) / f'(E_alt)

    Geometrisch: Man legt an der Stelle E die Tangente an die Kurve f und
    nimmt deren Nullstelle als bessere Schätzung. Meist reichen 3-5 Schritte.

    Parameter:
      tol      = Genauigkeit: Wenn die Korrektur kleiner als tol ist, hören wir auf.
      max_iter = Sicherheitsgrenze, damit die Schleife nie endlos läuft.
    """
    # Die Gleichung ist periodisch -> es genügt, M im Bereich [-pi, pi] zu lösen.
    M = wrap_pi(M)

    # Startwert der Iteration. Bei fast kreisförmigen Bahnen ist E ungefähr M.
    # Bei sehr exzentrischen Bahnen (e >= 0.8) startet man sicherheitshalber bei
    # +-pi, damit das Newton-Verfahren zuverlässig zur Lösung konvergiert.
    if e < 0.8:
        E = M
    else:
        E = math.pi if M >= 0 else -math.pi

    for _ in range(max_iter):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))   # f(E) / f'(E)
        E -= dE                                                    # Newton-Schritt
        if abs(dE) < tol:                                          # genau genug?
            break
    return E


def compute_orbit(altitude, speed, gamma_deg, phi_deg):
    """
    Berechnet aus den Startbedingungen alle Parameter der Bahn.

    Eingaben:
      altitude  = Höhe über der Erdoberfläche in km
      speed     = Startgeschwindigkeit in km/s
      gamma_deg = Richtung der Geschwindigkeit relativ zur Tangente in Grad.
                  0°  = genau tangential (parallel zur Erdoberfläche)
                  >0° = schräg von der Erde weg, <0° = schräg zur Erde hin
      phi_deg   = Startposition auf einem Kreis um die Erde in Grad
                  (0° = rechts von der Erde, 90° = oben, ...)

    Rückgabe: ein Dictionary (Wörterbuch) mit allen Ergebnissen.
      - Schlüssel "ok" = True, wenn die Bahn eine gültige Ellipse ist, die die
        Erde nicht schneidet. Dann stehen alle Bahnparameter im Dictionary.
      - Schlüssel "ok" = False, wenn nicht. Dann steht in "reason" eine
        Erklärung für den Benutzer, und nur die Startdaten sind gefüllt.
    """
    # ---- Schritt 1: Startzustand (Ort und Geschwindigkeit als Vektoren) ----
    r0 = EARTH_RADIUS + altitude       # Abstand vom ERDMITTELPUNKT, nicht von der Oberfläche
    gamma = math.radians(gamma_deg)    # Grad -> Bogenmaß
    phi = math.radians(phi_deg)

    # Zwei zueinander senkrechte Einheitsvektoren (Länge 1) am Startort:
    #   er = "radial": zeigt vom Erdmittelpunkt zum Satelliten
    #   et = "tangential": steht senkrecht darauf, gegen den Uhrzeigersinn
    er = np.array([math.cos(phi), math.sin(phi)])
    et = np.array([-math.sin(phi), math.cos(phi)])

    r_vec = r0 * er   # Ortsvektor des Satelliten (Erde = Ursprung)

    # Geschwindigkeitsvektor: Der Betrag "speed" wird auf die beiden Richtungen
    # aufgeteilt. cos(gamma) ist der Anteil entlang der Tangente, sin(gamma) der
    # Anteil von der Erde weg. Bei gamma = 0 ist die Geschwindigkeit rein tangential.
    v_vec = speed * (math.cos(gamma) * et + math.sin(gamma) * er)

    # Ergebnis-Dictionary: zunächst als "nicht ok" angelegt. Erst wenn alle
    # Prüfungen bestanden sind, wird "ok" auf True gesetzt.
    result = {
        "ok": False,
        "reason": "",
        "r0": r0,
        "r_vec": r_vec,
        "v_vec": v_vec,
        # Kreisbahngeschwindigkeit: Auf einer Kreisbahn ist die Gravitations-
        # beschleunigung mu/r^2 gleich der Zentripetalbeschleunigung v^2/r.
        # Daraus folgt v_c = sqrt(mu / r).
        "v_circ": math.sqrt(MU / r0),
        # Fluchtgeschwindigkeit: Setzt man die Energie gleich 0 (s. u.), folgt
        # v_esc = sqrt(2*mu/r) = sqrt(2) * v_c.
        "v_esc": math.sqrt(2.0 * MU / r0),
    }

    # ---- Schritt 2: Energie und Bahnform ---------------------------------------
    # Spezifische Energie (Energie pro Kilogramm) = kinetisch + potentiell:
    #     eps = v^2/2 - mu/r
    # Sie bleibt entlang der ganzen Bahn konstant (Energieerhaltung).
    #   eps < 0  -> gebundene Bahn (Ellipse oder Kreis)
    #   eps = 0  -> Parabel (genau Fluchtgeschwindigkeit)
    #   eps > 0  -> Hyperbel (der Satellit verlässt die Erde für immer)
    eps = speed ** 2 / 2.0 - MU / r0
    if eps >= 0:
        result["reason"] = (
            f"Die Geschwindigkeit von {fmt(speed, 2)} km/s ist nicht kleiner als die "
            f"Fluchtgeschwindigkeit ({fmt(result['v_esc'], 2)} km/s) in dieser Höhe. "
            "Damit ist die spezifische Energie ε ≥ 0: Die Bahn wäre eine Parabel oder "
            "Hyperbel, der Satellit würde die Erde verlassen. In dieser Version werden "
            "nur elliptische Bahnen dargestellt."
        )
        return result   # Abbruch: Alles Weitere gilt nur für Ellipsen.

    # ---- Schritt 3: Größe der Ellipse ------------------------------------------
    # Für gebundene Bahnen gilt eps = -mu / (2a). Umgestellt nach der großen
    # Halbachse a (halbe Länge der Ellipse an ihrer längsten Stelle):
    a = -MU / (2.0 * eps)

    # Spezifischer Drehimpuls h = x*vy - y*vx (das ist das Kreuzprodukt r x v in 2D).
    # Er bleibt konstant, weil die Gravitation immer zum Erdmittelpunkt zeigt und
    # deshalb kein Drehmoment erzeugt. Daraus folgt das 2. Kepler'sche Gesetz (Flächensatz).
    h = r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]

    # ---- Schritt 4: Form und Ausrichtung der Ellipse ---------------------------
    # Exzentrizitätsvektor:
    #     e_vec = [ (v^2 - mu/r) * r_vec - (r_vec · v_vec) * v_vec ] / mu
    # (Herleitung: e_vec = (v × h)/mu - r_vec/|r_vec| mit der Vektoridentität
    #  a × (b × c) = b (a·c) - c (a·b).)
    # Sein BETRAG ist die Exzentrizität e (0 = Kreis, nahe 1 = sehr lang gestreckt).
    # Seine RICHTUNG zeigt vom Erdmittelpunkt zum Perigäum (erdnächster Punkt).
    # np.dot(a, b) ist das Skalarprodukt zweier Vektoren.
    e_vec = ((speed ** 2 - MU / r0) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU
    e = float(np.linalg.norm(e_vec))   # Länge des Vektors; float() macht eine normale Zahl daraus
    e = min(e, 0.999999)               # Schutz: Durch Rundungsfehler darf e nie >= 1 werden,
                                       # sonst würde später sqrt(1 - e) abstürzen.

    # ---- Schritt 5: Perigäum und Apogäum -----------------------------------------
    # Der erdnächste Punkt hat den Abstand a*(1-e), der erdfernste a*(1+e)
    # (jeweils vom Erdmittelpunkt aus gemessen).
    r_p = a * (1.0 - e)   # Perigäum
    r_a = a * (1.0 + e)   # Apogäum

    # Liegt das Perigäum unter der Erdoberfläche, würde der Satellit abstürzen.
    if r_p < EARTH_RADIUS:
        # Tiefe unter der Oberfläche, mit Tausenderpunkt formatiert
        depth = fmt(EARTH_RADIUS - r_p, 0)
        result["reason"] = (
            f"Die Bahn schneidet die Erde: Das Perigäum liegt {depth} km unter der "
            "Erdoberfläche, der Satellit würde abstürzen. Erhöhe die Geschwindigkeit, "
            "ändere die Richtung der Geschwindigkeit oder wähle eine größere Höhe."
        )
        return result

    # ---- Schritt 6: Wo auf der Ellipse startet der Satellit? ---------------------
    # omega = Winkel der Perigäum-Richtung im Koordinatensystem.
    # atan2(y, x) liefert den Winkel eines Vektors (x, y) im richtigen Quadranten.
    # Sonderfall: Bei einer perfekten Kreisbahn (e ~ 0) gibt es kein eindeutiges
    # Perigäum. Dann legen wir es einfach in den Startpunkt (omega = phi).
    omega = math.atan2(e_vec[1], e_vec[0]) if e > 1e-10 else phi

    # Wahre Anomalie nu0 = Winkel zwischen Perigäum und Satellit, gesehen von der Erde.
    nu0 = wrap_pi(phi - omega)

    # Umrechnung wahre Anomalie -> exzentrische Anomalie E0 (Standardformel mit Halbwinkeln):
    #     E = 2 * atan2( sqrt(1-e) * sin(nu/2),  sqrt(1+e) * cos(nu/2) )
    E0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu0 / 2.0),
                          math.sqrt(1.0 + e) * math.cos(nu0 / 2.0))

    # Mittlere Anomalie zum Startzeitpunkt, direkt aus der Kepler-Gleichung:
    #     M0 = E0 - e * sin(E0)
    M0 = E0 - e * math.sin(E0)

    # Mittlere Bewegung n = sqrt(mu / a^3): Um wie viel Radiant pro Sekunde wächst M.
    n = math.sqrt(MU / a ** 3)

    # Umlaufzeit nach dem 3. Kepler'schen Gesetz: T = 2*pi*sqrt(a^3 / mu)
    T = 2.0 * math.pi * math.sqrt(a ** 3 / MU)

    # Alle Prüfungen bestanden -> Bahn ist gültig. Jetzt alle Ergebnisse ablegen.
    result.update(
        ok=True, eps=eps, a=a, e=e, h=h, omega=omega, M0=M0, n=n,
        r_p=r_p, r_a=r_a, T=T,
    )
    return result


def state_at(orbit, t):
    """
    Berechnet Ort und Geschwindigkeit des Satelliten t Sekunden nach dem Start.

    Parameter:
      orbit = das Dictionary aus compute_orbit (muss "ok" == True sein)
      t     = Zeit seit dem Start in Sekunden

    Rückgabe (5 Werte):
      pos = Ortsvektor [x, y] in km
      vel = Geschwindigkeitsvektor [vx, vy] in km/s
      r   = Abstand vom Erdmittelpunkt in km
      v   = Betrag der Geschwindigkeit in km/s
      nu  = wahre Anomalie in rad
    """
    a, e, w = orbit["a"], orbit["e"], orbit["omega"]

    # ---- Zeit -> mittlere Anomalie -> exzentrische Anomalie -----------------------
    # Die mittlere Anomalie wächst gleichmäßig: M(t) = M0 + n*t.
    # Die Kepler-Gleichung (numerisch gelöst) liefert daraus E.
    M = orbit["M0"] + orbit["n"] * t
    E = solve_kepler(M, e)

    cE, sE = math.cos(E), math.sin(E)   # cos(E) und sin(E) nur einmal berechnen
    r = a * (1.0 - e * cE)              # Abstand zur Erde: r = a * (1 - e*cos(E))
    b_fac = math.sqrt(1.0 - e * e)      # Verhältnis kleine/große Halbachse: b/a

    # ---- Ort in der Bahnebene ------------------------------------------------------
    # In einem Koordinatensystem, dessen x-Achse zum Perigäum zeigt, gilt:
    #     x = a * (cos(E) - e)
    #     y = a * sqrt(1 - e^2) * sin(E)
    # Das "- e" in x verschiebt die Ellipse so, dass die ERDE IM BRENNPUNKT liegt
    # (1. Kepler'sches Gesetz) und nicht im Mittelpunkt der Ellipse.
    xp, yp = a * (cE - e), a * b_fac * sE

    # ---- Geschwindigkeit in der Bahnebene -------------------------------------------
    # Ableitung des Ortes nach der Zeit (mit dE/dt = n / (1 - e*cos(E))):
    #     vx = -sqrt(mu*a)/r * sin(E)
    #     vy =  sqrt(mu*a*(1-e^2))/r * cos(E)
    k = math.sqrt(MU * a) / r
    vxp, vyp = -k * sE, k * b_fac * cE

    # ---- Drehung ins Koordinatensystem der Grafik ------------------------------------
    # Die Ellipse ist um den Winkel omega gedreht (Perigäum zeigt nicht nach +x).
    # Eine Drehung um den Winkel w wirkt auf einen Punkt (x, y) so:
    #     x' = x*cos(w) - y*sin(w)
    #     y' = x*sin(w) + y*cos(w)
    cw, sw = math.cos(w), math.sin(w)
    pos = np.array([xp * cw - yp * sw, xp * sw + yp * cw])
    vel = np.array([vxp * cw - vyp * sw, vxp * sw + vyp * cw])

    # Wahre Anomalie aus E (Umkehrung der Formel aus compute_orbit)
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                          math.sqrt(1.0 - e) * math.cos(E / 2.0))

    return pos, vel, r, float(np.linalg.norm(vel)), nu


def orbit_curve(orbit, n_points=721):
    """
    Liefert viele Punkte der gesamten Ellipse zum Zeichnen der Bahnkurve.

    Prinzip: Der Hilfswinkel E läuft einmal von 0 bis 2*pi. Für jedes E gibt es
    einen Punkt der Ellipse (gleiche Formeln wie in state_at). Mit 721 Punkten
    sieht die Kurve für das Auge völlig glatt aus.

    np.linspace(0, 2*pi, 721) erzeugt ein Array mit 721 gleichmäßig verteilten
    Werten; NumPy rechnet dann alle Punkte in einem Schritt aus (ohne Schleife).
    """
    a, e, w = orbit["a"], orbit["e"], orbit["omega"]
    E = np.linspace(0.0, 2.0 * np.pi, n_points)

    xp = a * (np.cos(E) - e)                           # Ellipse, Erde im Brennpunkt
    yp = a * math.sqrt(1.0 - e * e) * np.sin(E)

    # Gleiche Drehung um omega wie in state_at, hier für alle Punkte gleichzeitig.
    return (xp * math.cos(w) - yp * math.sin(w),
            xp * math.sin(w) + yp * math.cos(w))


def physics_checks(orbit):
    """
    Prüft, ob die Rechnung in sich stimmig ist (Selbsttest des Programms).

    Idee: In der Physik gibt es Größen, die sich entlang der Bahn NICHT ändern
    dürfen. Wir berechnen den Satelliten an mehreren Zeitpunkten und prüfen
    genau das:

      1. Energie:    eps = v^2/2 - mu/r          (Energieerhaltung)
      2. Drehimpuls: h = x*vy - y*vx             (2. Kepler'sches Gesetz)
      3. Startpunkt: Der berechnete Ort bei t = 0 muss der Startpunkt sein.
      4. Verhältnis T^2/a^3 muss 4*pi^2/mu sein  (3. Kepler'sches Gesetz).

    Wichtig: Energie und Drehimpuls werden in jedem Punkt NEU aus Ort und
    Geschwindigkeit berechnet, nicht aus den Startwerten übernommen. Nur dann
    ist es ein echter Test.

    Rückgabe: (max. rel. Abweichung Energie, max. rel. Abweichung Drehimpuls,
               Abstand zum Startpunkt in km, T^2/a^3, 4*pi^2/mu)
    """
    # 13 Zeitpunkte gleichmäßig über einen Umlauf (von t = 0 bis t = T)
    zeiten = np.linspace(0.0, orbit["T"], 13)

    d_eps, d_h = 0.0, 0.0   # größte bisher gefundene Abweichungen
    for t in zeiten:
        pos, vel, r, v, _ = state_at(orbit, float(t))   # "_" = unbenutzter Wert (nu)

        eps_t = v ** 2 / 2.0 - MU / r                    # Energie in diesem Punkt
        h_t = pos[0] * vel[1] - pos[1] * vel[0]          # Drehimpuls in diesem Punkt

        # Relative Abweichung = |neu - alt| / |alt|. Wir merken uns die größte.
        d_eps = max(d_eps, abs(eps_t - orbit["eps"]) / abs(orbit["eps"]))
        d_h = max(d_h, abs(h_t - orbit["h"]) / abs(orbit["h"]))

    # Stimmt der Ort bei t = 0 mit dem vorgegebenen Startpunkt überein?
    pos0, _, _, _, _ = state_at(orbit, 0.0)
    d_start = float(np.linalg.norm(pos0 - orbit["r_vec"]))

    # 3. Kepler'sches Gesetz: T^2 / a^3 = 4*pi^2 / mu
    ratio = orbit["T"] ** 2 / orbit["a"] ** 3
    ratio_ref = 4.0 * math.pi ** 2 / MU

    return d_eps, d_h, d_start, ratio, ratio_ref


# =============================================================================
# ABSCHNITT 5: DARSTELLUNG MIT PLOTLY (DIE GRAFIK)
# =============================================================================
def build_figure(orbit, pos, vel, show_traj):
    """
    Baut die Grafik: Erde, Bahn, Perigäum/Apogäum, Satellit, Fahrstrahl und
    Geschwindigkeitspfeil.

    Parameter:
      orbit     = Ergebnis von compute_orbit (gültig oder ungültig)
      pos, vel  = Ort und Geschwindigkeit, die eingezeichnet werden sollen
      show_traj = True -> die Bahnkurve (Ellipse) wird mitgezeichnet

    In Plotly besteht eine Grafik aus "Traces" (Linien, Punktwolken, Flächen),
    die nacheinander mit fig.add_trace(...) hinzugefügt werden. Später
    hinzugefügte Traces liegen über früheren.
    """
    fig = go.Figure()

    # ---- Die Erde als gefüllter Kreis -------------------------------------------------
    # Ein Kreis mit Radius R besteht aus den Punkten (R*cos(t), R*sin(t)) für
    # t von 0 bis 2*pi. Mit 361 Punkten (alle 1°) sieht er rund aus.
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    fig.add_trace(go.Scatter(
        x=EARTH_RADIUS * np.cos(theta),
        y=EARTH_RADIUS * np.sin(theta),
        mode="lines",                          # nur Linie (keine Punkte)
        fill="toself",                         # Fläche innerhalb der Linie füllen
        fillcolor="rgba(37,99,235,0.55)",      # Blau, 55 % deckend (rgba = rot, grün, blau, Deckkraft)
        line=dict(color="#60a5fa", width=2),   # hellblauer Rand
        name="Erde",                           # Text in der Legende
        hoverinfo="skip",                      # keine Maus-Info beim Darüberfahren
    ))

    if orbit["ok"]:
        # ---- Gültige Bahn ----------------------------------------------------------------
        # Der Plotbereich geht bis etwas über das Apogäum (15 % Rand), damit die
        # ganze Ellipse sichtbar ist. Alle Punkte der Ellipse liegen höchstens
        # r_a vom Ursprung entfernt, also passt ein Quadrat [-lim, lim].
        lim = 1.15 * orbit["r_a"]
        w = orbit["omega"]

        # Bahnkurve (nur wenn der Schalter "Bahnkurve anzeigen" an ist)
        if show_traj:
            ox, oy = orbit_curve(orbit)
            fig.add_trace(go.Scatter(
                x=ox, y=oy, mode="lines", name="Bahn",
                line=dict(color="#a78bfa", width=2.5), hoverinfo="skip",
            ))

        # Perigäum und Apogäum als beschriftete Marker.
        # Das Perigäum liegt in Richtung omega im Abstand r_p. Das Apogäum liegt
        # genau gegenüber (daher das Minuszeichen) im Abstand r_a.
        fig.add_trace(go.Scatter(
            x=[orbit["r_p"] * math.cos(w), -orbit["r_a"] * math.cos(w)],
            y=[orbit["r_p"] * math.sin(w), -orbit["r_a"] * math.sin(w)],
            mode="markers+text",                   # Punkt UND Text daneben
            text=["Perigäum", "Apogäum"],
            textposition="top center",
            marker=dict(size=9, symbol="diamond", color="#34d399"),
            textfont=dict(color="#34d399"),
            name="Perigäum / Apogäum",
            hoverinfo="skip",
        ))
    else:
        # ---- Ungültige Bahn ------------------------------------------------------------------
        # Es wird nur der Startpunkt gezeigt. Der Bereich ist ein Stück größer
        # als der Startabstand, damit der Geschwindigkeitspfeil Platz hat.
        lim = 1.3 * orbit["r0"]

    # ---- Fahrstrahl: gepunktete Linie von der Erde zum Satelliten ---------------------------------
    fig.add_trace(go.Scatter(
        x=[0.0, pos[0]], y=[0.0, pos[1]],
        mode="lines", name="Fahrstrahl",
        line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dot"),
        hoverinfo="skip",
    ))

    # ---- Der Satellit selbst: gelber Punkt mit weißem Rand --------------------------------------------
    fig.add_trace(go.Scatter(
        x=[pos[0]], y=[pos[1]], mode="markers", name="Satellit",
        marker=dict(size=14, color="#fbbf24", line=dict(color="white", width=1.5)),
    ))

    # ---- Geschwindigkeitsvektor als Pfeil -------------------------------------------------------------
    # Ein Pfeil ist in Plotly keine Trace, sondern eine "Annotation" und taucht
    # deshalb nicht von selbst in der Legende auf. Damit er trotzdem dort
    # erscheint, fügen wir eine unsichtbare Trace mit leeren Daten ([None]) und
    # der gleichen roten Farbe hinzu.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name="Geschwindigkeitsvektor",
        line=dict(color="#f87171", width=3),
    ))

    speed = float(np.linalg.norm(vel))
    if speed > 0:
        u = vel / speed          # Einheitsvektor in Richtung der Geschwindigkeit
        length = 0.16 * lim      # Pfeillänge = 16 % der Achsenweite. Sie ist immer gleich lang:
                                 # der Pfeil zeigt nur die RICHTUNG, der Betrag steht rechts in den Ergebnissen.
        fig.add_annotation(
            # x, y = Pfeilspitze; ax, ay = Pfeilende ("Schwanz") beim Satelliten
            x=pos[0] + length * u[0], y=pos[1] + length * u[1],
            ax=pos[0], ay=pos[1],
            # "x"/"y" bedeutet: Koordinaten sind Datenkoordinaten der Achsen (km),
            # nicht Pixel. Deshalb bleibt der Pfeil beim Zoomen am Satelliten.
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.5,
            arrowcolor="#f87171", text="",
        )

    # ---- Aussehen der Grafik --------------------------------------------------------------------------------
    grid = "rgba(255,255,255,0.07)"   # sehr helles, durchscheinendes Gitter
    fig.update_layout(
        template="plotly_dark",        # dunkles Grunddesign
        paper_bgcolor="#0b1020",       # Hintergrund außerhalb des Diagramms
        plot_bgcolor="#0b1020",        # Hintergrund im Diagramm
        height=700,
        margin=dict(l=10, r=10, t=10, b=10),   # kleine Ränder
        xaxis=dict(range=[-lim, lim], title="x (km)", zeroline=False, gridcolor=grid),
        # scaleanchor="x" + scaleratio=1: 1 km in y-Richtung ist genauso lang wie
        # 1 km in x-Richtung. Ohne das würde die Ellipse verzerrt und die Erde
        # wie ein Oval aussehen.
        yaxis=dict(range=[-lim, lim], title="y (km)", zeroline=False, gridcolor=grid,
                   scaleanchor="x", scaleratio=1),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(11,16,32,0.6)"),   # Legende oben links
    )
    return fig


# =============================================================================
# ABSCHNITT 6: CALLBACK-FUNKTIONEN FÜR DIE KNÖPFE
# =============================================================================
# Ein "Callback" ist eine Funktion, die Streamlit aufruft, sobald ein Knopf
# gedrückt wurde, und zwar VOR dem erneuten Ausführen des Skripts. Hier dürfen
# wir deshalb auch die Werte von Reglern ändern (z. B. die Zeit auf 0 setzen).
# Außerhalb von Callbacks ist das für bereits gezeichnete Regler nicht erlaubt.
#
# Die Werte liegen in st.session_state, einem Speicher, der zwischen den
# Neustarts des Skripts erhalten bleibt (siehe Abschnitt 7).
def cb_start():
    """Start-Knopf: Animation einschalten."""
    st.session_state.running = True


def cb_pause():
    """Pause-Knopf: Animation anhalten (die Zeit bleibt stehen)."""
    st.session_state.running = False


def cb_reset():
    """Reset-Knopf: Animation anhalten und Zeit zurück auf 0."""
    st.session_state.running = False
    st.session_state.t_frac = 0.0


def cb_circular():
    """
    Stellt die Kreisbahngeschwindigkeit für die aktuell gewählte Höhe ein und
    setzt die Richtung auf 0° (tangential), denn nur dann ist die Bahn ein Kreis.

    Der Wert wird auf 2 Nachkommastellen gerundet, weil der Regler in 0,01-km/s-
    Schritten arbeitet. Deshalb ist die Bahn danach "fast" ein Kreis (e ~ 0,0007).
    """
    r0 = EARTH_RADIUS + st.session_state.altitude
    st.session_state.speed = round(math.sqrt(MU / r0), 2)
    st.session_state.gamma = 0


# =============================================================================
# ABSCHNITT 7: BENUTZEROBERFLÄCHE MIT STREAMLIT (DER HAUPTABLAUF)
# =============================================================================
#
# WICHTIG ZUM VERSTÄNDNIS VON STREAMLIT:
# Streamlit führt dieses gesamte Skript bei JEDER Eingabe (Regler bewegt, Knopf
# gedrückt) komplett von oben nach unten NEU aus. Normale Python-Variablen sind
# danach wieder weg. Was erhalten bleiben soll, steht in st.session_state, einem
# Speicher, der über die Neustarts hinweg bestehen bleibt.
#
# Jeder Regler hat einen "key". Unter diesem Namen liegt sein aktueller Wert in
# st.session_state. Beispiel: st.slider(..., key="altitude") -> der Wert steht
# in st.session_state.altitude.
# -----------------------------------------------------------------------------

# Muss der erste Streamlit-Befehl des Skripts sein: Titel des Browser-Tabs,
# Icon und breites Layout (nutzt die ganze Bildschirmbreite).
st.set_page_config(page_title="Satellite Orbit Simulator", page_icon="🛰️", layout="wide")

# ---- Startwerte -----------------------------------------------------------------------
# Das sind die Werte beim allerersten Start. "mass" ist eine Kommazahl (500.0),
# weil das Eingabefeld unten mit Kommazahlen als Grenzen arbeitet (die Typen
# müssen zusammenpassen).
DEFAULTS = {
    "mass": 500.0,           # Masse des Satelliten in kg
    "altitude": 400,         # Höhe in km
    "speed": 7.67,           # Startgeschwindigkeit in km/s
    "gamma": 0,              # Richtung der Geschwindigkeit in Grad (0 = tangential)
    "phi": 0,                # Startposition in Grad
    "t_frac": 0.0,           # Zeit in Umlaufzeiten
    "running": False,        # läuft die Animation gerade?
    "show_traj": True,       # Bahnkurve anzeigen?
    "anim_speed": "Normal",  # Animationsgeschwindigkeit
}
for _key, _val in DEFAULTS.items():
    # setdefault trägt den Wert nur ein, wenn der Schlüssel noch NICHT existiert.
    # Beim ersten Start werden so die Startwerte gesetzt, bei allen späteren
    # Neustarts bleiben die Werte des Benutzers erhalten.
    st.session_state.setdefault(_key, _val)

# ---- Ein Animationsschritt (falls die Animation läuft) ---------------------------------------
# Wenn "Start" gedrückt wurde, rückt die Zeit bei jedem Neustart des Skripts um
# einen kleinen Schritt vor. Das muss HIER passieren, bevor der Zeit-Regler
# weiter unten gezeichnet wird: Streamlit erlaubt es nicht, den Wert eines
# Reglers zu ändern, nachdem er im selben Durchlauf schon gezeichnet wurde.
if st.session_state.running:
    _new_t = st.session_state.t_frac + ANIM_STEPS[st.session_state.anim_speed]
    if _new_t > T_MAX:
        _new_t -= T_MAX                          # nach 2 Umläufen wieder von vorn beginnen
    st.session_state.t_frac = round(_new_t, 3)   # runden, damit keine Rundungsreste (0.30000000004) entstehen

# ---- Überschrift und Spaltenaufteilung ------------------------------------------------------------
st.title("🛰️ Satellite Orbit Simulator")
st.caption("2D · Keplersche Gesetze · Python")

# Zwei Spalten im Verhältnis 3 : 2. Links kommt die Grafik, rechts die Bedienung.
# Mit "with col_xyz:" legt man fest, in welcher Spalte die folgenden Elemente erscheinen.
col_plot, col_ctrl = st.columns([3, 2], gap="large")

# ---- Rechte Spalte: Parameter und Bedienelemente ------------------------------------------------------
with col_ctrl:
    st.subheader("Parameter")

    # Masse: hat KEINEN Einfluss auf die Bahn (siehe Kommentar bei MU), wird nur
    # für die Anzeige der Gesamtenergie benutzt.
    st.number_input("Masse des Satelliten (kg)", min_value=1.0, max_value=100000.0,
                    step=50.0, format="%.0f", key="mass")

    # Die Regler: st.slider(Beschriftung, Minimum, Maximum, step=Schrittweite, key=...)
    st.slider("Höhe über der Erdoberfläche (km)", 200, 36000, step=50, key="altitude")
    st.slider("Startgeschwindigkeit (km/s)", 1.0, 11.5, step=0.01, format="%.2f", key="speed")
    st.slider("Richtung der Geschwindigkeit (°), 0° = tangential", -90, 90, step=1, key="gamma")
    st.slider("Startposition (°)", 0, 359, step=1, key="phi")

    # on_click=... gibt die Funktion an, die beim Klick ausgeführt wird (Callback).
    st.button("Kreisbahngeschwindigkeit einstellen", on_click=cb_circular)

    st.divider()   # waagerechte Trennlinie

    # Drei Knöpfe nebeneinander: Dafür teilen wir die Spalte in drei Mini-Spalten.
    b1, b2, b3 = st.columns(3)
    b1.button("▶ Start", on_click=cb_start)
    b2.button("⏸ Pause", on_click=cb_pause)
    b3.button("⟲ Reset", on_click=cb_reset)

    st.toggle("Bahnkurve anzeigen", key="show_traj")
    st.select_slider("Animationsgeschwindigkeit", options=list(ANIM_STEPS), key="anim_speed")

    # Zeit-Regler. Wichtig: Sein Wert steht in st.session_state.t_frac und wird
    # von der Animation (oben) verändert. Deshalb bewegt sich der Regler während
    # der Animation von selbst.
    st.slider("Zeit seit Simulationsbeginn (in Umlaufzeiten T)", 0.0, T_MAX,
              step=0.001, format="%.3f", key="t_frac")

# ---- Berechnung ----------------------------------------------------------------------------------------------
# Bei jedem Durchlauf wird die Bahn aus den aktuellen Reglerwerten neu berechnet.
# Das ist schnell genug (wenige Mikrosekunden), deshalb braucht man keinen Zwischenspeicher.
orbit = compute_orbit(
    st.session_state.altitude, st.session_state.speed,
    st.session_state.gamma, st.session_state.phi,
)

if orbit["ok"]:
    # Der Zeit-Regler zählt in Umlaufzeiten, state_at erwartet aber Sekunden.
    t_sec = st.session_state.t_frac * orbit["T"]
    pos, vel, r_now, v_now, nu_now = state_at(orbit, t_sec)
else:
    # Bahn ungültig: Wir zeigen den Satelliten im Startpunkt mit dem
    # Geschwindigkeitspfeil. So sieht man, was an der Eingabe nicht passt.
    t_sec, pos, vel = 0.0, orbit["r_vec"], orbit["v_vec"]

# ---- Linke Spalte: Grafik ---------------------------------------------------------------------------------------
with col_plot:
    st.plotly_chart(build_figure(orbit, pos, vel, st.session_state.show_traj))

# ---- Rechte Spalte (unterer Teil): Ergebnisse ---------------------------------------------------------------------
# Ein zweiter "with col_ctrl:"-Block hängt seine Elemente UNTER die Regler an.
with col_ctrl:
    st.divider()
    st.subheader("Ergebnisse")

    # Fallbeschleunigung in dieser Höhe: g = mu / r^2 (km/s^2); mal 1000 ergibt m/s^2.
    g_here = MU / orbit["r0"] ** 2 * 1000.0

    # Zwei Spalten für die Kennzahlen. st.metric(Titel, Wert) zeigt eine große Zahl.
    c1, c2 = st.columns(2)
    c1.metric("Kreisbahngeschwindigkeit", f"{fmt(orbit['v_circ'], 3)} km/s")
    c2.metric("Fluchtgeschwindigkeit", f"{fmt(orbit['v_esc'], 3)} km/s")
    st.caption(
        f"Fallbeschleunigung in dieser Höhe: {fmt(g_here, 2)} m/s² "
        f"({fmt(100 * g_here / G0, 0)} % des Werts an der Erdoberfläche). "
        "Der Satellit ist also nicht „schwerelos“, sondern befindet sich im "
        "freien Fall um die Erde."
    )

    if not orbit["ok"]:
        # Ungültige Bahn: rote Fehlermeldung mit der Erklärung aus compute_orbit.
        st.error(orbit["reason"])
    else:
        # Kleine Einordnung der Bahnform (e < 0,01 gilt hier als "nahezu kreisförmig").
        if orbit["e"] < 0.01:
            st.success("Die Bahn ist nahezu kreisförmig.")
        else:
            st.info("Die Bahn ist elliptisch.")

        # Die Bahnparameter. Perigäum und Apogäum werden als HÖHE über der
        # Oberfläche angezeigt, deshalb wird der Erdradius abgezogen.
        c1.metric("Exzentrizität", fmt(orbit["e"], 4))
        c2.metric("Große Halbachse", f"{fmt(orbit['a'], 0)} km")
        c1.metric("Perigäum (Höhe)", f"{fmt(orbit['r_p'] - EARTH_RADIUS, 0)} km")
        c2.metric("Apogäum (Höhe)", f"{fmt(orbit['r_a'] - EARTH_RADIUS, 0)} km")
        c1.metric("Umlaufzeit", f"{fmt(orbit['T'] / 60, 2)} min")
        c2.metric("Aktuelle Geschwindigkeit", f"{fmt(v_now, 3)} km/s")
        c1.metric("Aktuelle Höhe", f"{fmt(r_now - EARTH_RADIUS, 0)} km")
        c2.metric("Zeit t", f"{fmt(t_sec / 60, 2)} min")

        # Die Masse beeinflusst die Bahn nicht, aber die Gesamtenergie E = m * eps.
        # Einheiten: kg * km^2/s^2 = 10^6 kg*m^2/s^2 = 1 Megajoule (MJ).
        e_total = st.session_state.mass * orbit["eps"]
        st.caption(
            "Die Masse des Satelliten beeinflusst weder Form noch Umlaufzeit der "
            f"Bahn (sie geht nur in die Gesamtenergie ein: {fmt(e_total, 0)} MJ)."
        )

        # Aufklappbarer Bereich mit dem Selbsttest aus physics_checks.
        with st.expander("Überprüfung der Physik (Kepler'sche Gesetze)"):
            d_eps, d_h, d_start, ratio, ratio_ref = physics_checks(orbit)
            st.write(f"**3. Kepler'sches Gesetz:** T²/a³ = {fmt_sci(ratio, 6)} s²/km³, "
                     f"4π²/μ = {fmt_sci(ratio_ref, 6)} s²/km³")
            st.write(f"**Energieerhaltung** (größte relative Abweichung über einen Umlauf): "
                     f"{fmt_sci(d_eps, 2)}")
            st.write(f"**Drehimpulserhaltung** (2. Kepler'sches Gesetz): {fmt_sci(d_h, 2)}")
            st.write(f"**Übereinstimmung mit dem Startpunkt:** {fmt_sci(d_start, 2)} km")
            st.caption("Werte in der Größenordnung 1e-10 und kleiner sind reine "
                       "Rundungsfehler: Die Rechnung ist in sich stimmig.")

# ---- Animation: nächstes Bild anfordern -------------------------------------------------------------------------------
# Läuft die Animation, warten wir kurz (0,05 s, damit der Browser das Bild
# anzeigen kann) und starten das Skript sofort neu. Beim Neustart rückt die Zeit
# oben um einen Schritt vor, und die Grafik wird mit der neuen Satellitenposition
# gezeichnet. So entsteht Bild für Bild die Bewegung. Das Pausieren funktioniert,
# weil cb_pause "running" auf False setzt und dann keine Neustarts mehr ausgelöst werden.
if st.session_state.running and orbit["ok"]:
    time.sleep(0.05)
    st.rerun()
