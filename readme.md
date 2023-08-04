# Ready for Robots Prototypes

Buch unter [https://kolumdium.github.io/r4r-prototypes/book/intro.html](https://kolumdium.github.io/r4r-prototypes/book/intro.html)

## Local Development

### Erzeugen der Umgebung

Das File ist explicit für win-64 erstellt worden. Für andere Systeme muss das File `geo-env-all.txt` verwendet werden.

```shell
conda env create --file geo-env.txt
```

### Neuen Content adden

Neue Notebooks unter prototypes erstellen und in `book/_toc.yml` eintragen. Danach das Buch neu bauen.

### Buch bauen

```shell
jupyter-book build .
```

### Buch pushen

Wenn build erfolgreich war, dann kann das Buch lokal unter `_build/html/index.html` betrachtet werden.
Falls zufrieden mit dem Ergebniss, dann kann das Buch gepusht werden.

```shell
ghp-import -n -p -f _build/html
```
