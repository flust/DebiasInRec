# Data Processing

1. filter.py

- Input: train.csv, song.csv, members.csv
- Output: top_song.csv, listener.csv


2. Convert top_song.csv to data format

- Script: top_song_ffm.py top_song_fm.py top_song_mf.py 
- Input: top_song.csv
- Output: top_song.{ffm, fm, mf}

3. Convert listener.csv to data format

- Script: listener_ffm.py listener_fm.py listener_mf.py 
- Input: listener.csv
- Output: listener.{ffm, fm, mf}

4. Split listener.{ffm, fm, mf} to tr va te

- Script: split.sh
- Input: listener.{ffm, fm, mf}
- Output: listener.{tr, va, tr}.{ffm, fm, mf}

