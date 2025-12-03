import gymnasium as gym
import ale_py
import numpy as np
import cv2

gym.register_envs(ale_py)

def find_pellet_color():
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    obs, _ = env.reset()
    
    # Skip some frames to let the game "warm up" and show sprites
    for _ in range(50):
        obs, _, _, _, _ = env.step(0)

    # Crop to the play area (remove score at bottom/top)
    # We use a small slice where we KNOW pellets usually are (near middle-ish)
    # y: 30 to 50, x: 20 to 140
    slice_img = obs[30:60, 20:140]
    
    # Reshape to list of pixels
    pixels = slice_img.reshape(-1, 3)
    
    # Find unique colors and their counts
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # Sort by most frequent
    sorted_indices = np.argsort(-counts)
    
    print("="*40)
    print("TOP 5 COLORS DETECTED IN GAME AREA")
    print("="*40)
    print("Use the RGB value that looks like a Pellet color")
    print("(Usually NOT [0 0 0] (Black) or [28 28 127] (Blue Walls))")
    print("-" * 40)
    
    for i in range(min(10, len(colors))):
        rgb = colors[sorted_indices[i]]
        count = counts[sorted_indices[i]]
        print(f"Color: {rgb} | Pixel Count: {count}")
        
        # Heuristic guess
        if np.mean(rgb) > 0 and not (rgb[0] < 50 and rgb[1] < 50): # Not black, not dark blue
             print(f"   ^^^ LIKELY PELLET CANDIDATE ^^^")

    env.close()

if __name__ == "__main__":
    find_pellet_color()
