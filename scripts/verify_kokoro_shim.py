import sys
import os
sys.path.append(os.getcwd()) # Ensure local misaki is found

try:
    from kokoro import KPipeline
    print("✅ Kokoro imported.")
    pipeline = KPipeline(lang_code='a', device='cpu')
    print("✅ KPipeline instantiated.")
    
    # Test generation
    gen = pipeline("Hello, this is a test.", voice="am_michael", speed=1)
    for res in gen:
        print("✅ Generated audio chunk.")
        break
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
