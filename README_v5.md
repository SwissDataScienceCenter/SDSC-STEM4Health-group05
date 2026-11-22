# v5 auf RenkuLab: hochladen, laufen lassen, einreichen

`final_v5.py` ist euer `final.py`, erweitert um eine **Sollwert-Fenster-Regel** für Prozess A. Prozess B ist unverändert. Das Skript reicht **nichts** ein.

## Was neu ist (Prozess A)
Ein Batch wird abgelehnt, wenn **eine** der beiden Regeln ablehnt:
1. **Sollwert-Fenster** (lernt nur aus guten Batches): Der Median jedes Sensors muss im Bereich der guten Batches liegen, plus 50 % Spielraum. So werden klare Sollwert-Verschiebungen abgefangen, kleine akzeptable Verschiebungen aber toleriert.
2. **Ensemble** wie bisher: PCA-Score und Extra-Trees, Grenze auf den F1 abgestimmt.

Weggefallen ist die feste PCA-95-%-Grenze als eigene Regel. In der Kreuzvalidierung hat sie vor allem Fehlalarme bei akzeptablen Sollwert-Verschiebungen erzeugt. PCA steckt weiterhin im Ensemble.

| Prozess A, Kreuzvalidierung 5 × 5 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| pca_or_ensemble (letzte Submission) | 0.862 | 0.914 | 0.903 | 0.908 |
| **window_or_ensemble (v5)** | **0.880** | **0.916** | **0.926** | **0.920** |

Auf dem Testsatz ändern sich 5 von 161 Batches gegenüber der letzten Submission:
- **neu abgelehnt:** A_043
- **nicht mehr abgelehnt:** A_120, A_147, A_165, A_216

Insgesamt werden 34 statt 37 Batches abgelehnt.

## Schritt 1: Hochladen
`final_v5.py` und `v5_RenkuLab.ipynb` in **denselben Ordner wie euer `final.py`** legen (`scripts/`). Die Daten findet das Skript selbst, genau wie bei `final.py`.

## Schritt 2: Laufen lassen
`v5_RenkuLab.ipynb` öffnen:
- **Zelle 1** (etwa 1–2 Minuten): erstellt die Einreichungsdatei ohne Validierung. Erwartet am Ende: A 127 accepted / 34 rejected, B 35 / 5. Dazu die Zeile `newly rejected ['A_043'], no longer rejected ['A_120', 'A_147', 'A_165', 'A_216']`.
- **Zelle 2** (optional, etwa 10 Minuten): dasselbe mit Kreuzvalidierung, also mit Kennzahlen.
- **Zelle 3** (optional): die vorsichtige Variante `window_or_current`, also die letzte Submission plus A_043 (A: 123 / 38).

Prozess B bleibt bei F1 0.999 in der Kreuzvalidierung. Validierter Gesamtscore: 0.960 (bisher 0.954).

Ergebnisse in `scripts/results/`:
- `submission_v5_A-<modell>.csv`: Einreichungsdatei
- `test_A_vergleich.csv`: pro Test-Batch die alte und die neue Entscheidung
- `history/` und `submission_history.csv`: Verlauf aller Läufe

## Schritt 3: Einreichen (im Leaderboard-Notebook)
```python
%run /pfad/zu/scripts/final_v5.py --skip-cv
print(submission.groupby(["project", "label"]).size())   # Kontrolle: A 127/34, B 35/5
metrics = submit(submission, user="Team XX")              # derselbe Teamname wie bisher
metrics
```
Ohne Neuberechnung geht es auch so:
```python
import pandas as pd
submission = pd.read_csv("/pfad/zu/scripts/results/submission_v5_A-window_or_ensemble.csv", dtype=str)
metrics = submit(submission, user="Team XX")
```

## Leaderboard-Ergebnis deuten
- **A-F1 steigt** über 0.934: v5 behalten.
- **A-F1 sinkt:** Die 4 PCA-Ablehnungen waren eher richtig. Dann `window_or_current` einreichen.

## Wenn etwas schiefgeht
- **`BRPMM_template.csv not found`**: `final_v5.py` liegt nicht im selben Ordner wie `final.py`.
- **`ModuleNotFoundError`**: `!pip install pandas numpy scikit-learn` ausführen.
- **Andere Zahlen als oben:** nicht einreichen, sondern die Ausgabe an Claude schicken.
