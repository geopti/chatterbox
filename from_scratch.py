from chatterbox_pytorch.tts import ChatterboxTTS

# Load model
tts = ChatterboxTTS.from_pretrained('cpu')  # or 'cuda'

# Generate speech
wav = tts.generate('Hello what are you doing ? All is good , my friend')

# Save
import scipy.io.wavfile as wavfile
wavfile.write('output.wav', tts.sr, wav.squeeze().numpy())