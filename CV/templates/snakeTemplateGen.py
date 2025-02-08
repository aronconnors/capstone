import numpy as np
from PIL import Image

def create_square_with_border(inner_color, border_color, inner_size, border_size, filename='square_with_border.png'):
    """
    Generates an image with a solid border.

    Parameters:
    - inner_color: Tuple of RGB values for the inner square (e.g., (0, 0, 255)).
    - border_color: Tuple of RGB values for the border (e.g., (255, 204, 0)).
    - inner_size: Size (width and height in pixels) of the inner square.
    - border_size: Width (in pixels) of the border around the inner square.
    - filename: The filename to save the image.
    """
    # Calculate the full image dimensions.
    full_size = inner_size + 2 * border_size

    # Create an empty image filled with the border color.
    img = np.full((full_size, full_size, 3), border_color, dtype=np.uint8)

    # Fill the inner square with the inner color.
    img[border_size:border_size+inner_size, border_size:border_size+inner_size] = inner_color

    # Convert the NumPy array to a PIL image.
    im = Image.fromarray(img, 'RGB')
    im.save(filename)
    im.show()

# Example usage:
if __name__ == "__main__":
    # Define colors as RGB tuples.
    RED1 = (200, 0, 0)
    RED2 = (255, 50, 50)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    YELLOW1 = (255, 204, 0)
    YELLOW2 = (255, 255, 102)
    
    # Specify inner square size and border width.
    inner_size = 20
    border_size = 3
    
    # Create the square with a border.
    create_square_with_border(inner_color=RED2, border_color=RED1,
        inner_size=inner_size, border_size=border_size,
        filename='food.png')
    
    create_square_with_border(inner_color=YELLOW2, border_color=YELLOW1,
        inner_size=inner_size, border_size=border_size,
        filename='head.png')
