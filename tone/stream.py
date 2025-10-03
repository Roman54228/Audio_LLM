from tone import StreamingCTCPipeline, read_stream_example_audio


pipeline = StreamingCTCPipeline.from_local("/models")

state = None  # Current state of the ASR pipeline (None - initial)
for audio_chunk in read_stream_example_audio():  # Use any source of audio chunks
    new_phrases, state = pipeline.forward(audio_chunk, state)
    print(new_phrases)

# Finalize the pipeline and get the remaining phrases
new_phrases, _ = pipeline.finalize(state)
print(new_phrases)

