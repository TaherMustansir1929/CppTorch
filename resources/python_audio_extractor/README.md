# Python Audio Features Extractor Script

### Install dependencies:
```python
python3 -m pip install -r requirements.txt
```

### Run script:
```python
python3 gender_audio_extractor.py \
  --male-dir audio/male \
  --female-dir audio/female \
  --output manually_extracted_gender_audio_features.csv \
  --workers 4 \
  --male-label 0 \
  --female-label 1
```