import os
import colorama


def display_logo():

    # Load the logo.
    logo_path = os.path.join("assets", "asciilogo.txt")
    if not os.path.exists(logo_path):
        raise FileNotFoundError("The logo file is missing.")
    with open(logo_path, "r") as f:
        logo = f.read()

    # Print the logo line by line. Use colorama to colorize the output. Use a cyberpunk color scheme.
    for line_index, line in enumerate(logo.split("\n")):
        color = colorama.Fore.GREEN
        style = colorama.Style.BRIGHT if line_index % 2 == 0 else colorama.Style.NORMAL
        print(color + style + line)
    print(colorama.Style.RESET_ALL)