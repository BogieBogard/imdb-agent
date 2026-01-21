class MToken:
    def __init__(self, text, phonemes):
        self.text = text
        self.phonemes = phonemes
        self.whitespace = " " # Dummy whitespace to be safe

class G2P:
    def __init__(self, **kwargs):
        # Accept any arguments kokoro throws at us (trf, british, unk, etc)
        self.lang = 'en-gb' if kwargs.get('british') else 'en-us'

    def __call__(self, text):
        from phonemizer import phonemize
        # Use espeak to get IPA phonemes. preserving punctuation might be needed?
        # Kokoro expects tokens. We'll verify if one big token works.
        try:
            # espeak backend is fast. strip=True removes end newlines.
            # with_stress=True? Kokoro uses phonemes.
            ps = phonemize(text, language=self.lang, backend='espeak', strip=True)
        except Exception as e:
            print(f"DEBUG: Phonemizer failed: {e}")
            ps = text # Fallback
            
        # Return as one big token for now
        # Kokoro expects (something, tokens) tuple based on unpacking error
        return (None, [MToken(text, ps)])
