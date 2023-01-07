from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")

# 4. apply pretrained pipeline
diarization = pipeline("/home/nttung/research/Monash_CCU/mock_data/audio_data/M01000FO1.wav")

# 5. print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")