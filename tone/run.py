from tone import StreamingCTCPipeline, read_audio, read_example_audio


audio = read_example_audio() # or read_audio("your_audio.flac")
audio = read_audio("ml_audi.wav")
# pipeline = StreamingCTCPipeline.from_hugging_face()
pipeline = StreamingCTCPipeline.from_local("/models")

print(pipeline.forward_offline(audio))  # run offline recognition
