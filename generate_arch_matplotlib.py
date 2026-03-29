import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_cnn_architecture():
    # Define the layers
    layers = [
        {"name": "Input Layer", "details": "(Batch, 17, 5, 17)", "color": "#E3F2FD"},
        {"name": "Conv2D (32, 3x3) + BatchNorm", "details": "Activation: ReLU", "color": "#FFF3E0"},
        {"name": "MaxPooling2D (2x2)", "details": "Downsample", "color": "#FFCCBC"},
        {"name": "Conv2D (64, 3x3) + BatchNorm", "details": "Activation: ReLU", "color": "#FFF3E0"},
        {"name": "MaxPooling2D (2x2)", "details": "Downsample", "color": "#FFCCBC"},
        {"name": "Conv2D (128, 3x3) + BatchNorm", "details": "Activation: ReLU", "color": "#FFF3E0"},
        {"name": "Global Average Pooling", "details": "Flattening", "color": "#E8F5E9"},
        {"name": "Dense (128) + Dropout (0.3)", "details": "Activation: ReLU", "color": "#F3E5F5"},
        {"name": "Dense (64)", "details": "Activation: ReLU", "color": "#F3E5F5"},
        {"name": "Dense Output (1)", "details": "Predicted Price", "color": "#ECEFF1"}
    ]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.axis('off')

    y_pos = 9.5
    box_height = 0.6
    box_width = 3.5
    x_pos = 1.0

    for i, layer in enumerate(layers):
        # Draw Box
        rect = patches.Rectangle(
            (x_pos, y_pos), box_width, box_height, 
            linewidth=1.2, edgecolor='#333333', facecolor=layer["color"]
        )
        ax.add_patch(rect)
        
        # Add Text Inside
        ax.text(x_pos + box_width/2, y_pos + box_height/2 + 0.1, layer["name"], 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=10, fontweight='bold', color='#212121')
        ax.text(x_pos + box_width/2, y_pos + box_height/2 - 0.15, layer["details"], 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=8, color='#424242')

        # Draw Arrow
        if i < len(layers) - 1:
            ax.annotate("", 
                        xy=(x_pos + box_width/2, y_pos - 0.3), 
                        xytext=(x_pos + box_width/2, y_pos), 
                        arrowprops=dict(arrowstyle="->", lw=1.5, color='#333333'))

        y_pos -= 1.0

    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 10.5)
    plt.title("CNN Model Architecture", fontsize=14, fontweight='bold', pad=20)
    
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/fig4_cnn_arch.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"CNN Architecture diagram successfully generated at {out_path}!")

if __name__ == '__main__':
    draw_cnn_architecture()
