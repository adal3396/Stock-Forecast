import os
import numpy as np
import visualkeras
from model import build_cnn, compile_model

def main():
    print("Loading test dataset shape for model scaling...")
    X_te = np.load("X_test.npy")
    input_shape = X_te.shape[1:]
    
    print("Constructing architecture model...")
    arch_model = build_cnn(input_shape)
    compile_model(arch_model)
    
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/fig4_cnn_arch.png"
    print(f"Generating diagram with visualkeras to {out_path}...")
    
    # Try using AggFont if available to render legend nicely
    font_kwargs = {}
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 16)
        font_kwargs['font'] = font
    except Exception:
        pass
        
    visualkeras.layered_view(arch_model, to_file=out_path, legend=True, **font_kwargs)
    print("CNN Architecture diagram successfully generated!")

if __name__ == "__main__":
    main()
