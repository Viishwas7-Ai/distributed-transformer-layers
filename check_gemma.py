# check_gemma.py
import inspect
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
from transformers import AutoConfig

print("--- Running GemmaRotaryEmbedding Inspector ---")

try:
    # This line will get the exact signature of the __init__ method
    signature = inspect.signature(GemmaRotaryEmbedding.__init__)
    
    print("\n[SUCCESS] Successfully inspected GemmaRotaryEmbedding.__init__")
    print("The arguments for your installed version are:")
    
    # Print the list of all found arguments
    print(list(signature.parameters.keys()))

except Exception as e:
    print(f"\n[ERROR] An error occurred during inspection: {e}")

print("\n--- Inspection complete ---")