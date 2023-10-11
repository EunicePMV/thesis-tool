import PySimpleGUI as sg
import io

# Define the home layout with updated settings
home_layout = [
    [sg.Text("Welcome to the Image Analyzer", font=("Any", 30), justification="center")],  # Increase font size
    [sg.Column(layout=[[sg.Button("Start", size=(20, 3), font=("Any", 20))]], justification='c')]  # Increase button size and font
]

# Define the second layout
second_layout = [
    [sg.Text("Upload an image:", font=("Any", 18), justification="center")],  # Increase font size
    [sg.InputText(key="-IMAGE_PATH-"), sg.FileBrowse(file_types=(("Images", "*.png *.jpg *.jpeg *.gif"),), size=(20, 1))],
    [sg.Button("Upload", font=("Any", 18))],  # Increase button font
    [sg.Image(key="-IMAGE-", size=(300, 300))],
    [sg.Text("Analysis Result:", font=("Any", 18), justification="center")],  # Increase font size
    [sg.Text("", size=(30, 1), key="-RESULT-", justification="center", font=("Any", 18))],  # Increase font size
]

# Create the layout that includes both home and second layouts
layout = [
    [sg.Column(home_layout, key="-HOME-", visible=True, justification="center")],  # Center the column
    [sg.Column(second_layout, key="-SECOND-", visible=False, justification="center")],  # Center the column
]

# Create the main window with the specified size
window = sg.Window("Image Analyzer", layout, finalize=True, size=(750, 750))

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event == "Start":
        # Switch to the second layout
        window["-HOME-"].update(visible=False)
        window["-SECOND-"].update(visible=True)

    if event == "Upload":
        image_path = values["-IMAGE_PATH-"]
        
        if image_path:
            try:
                # Open and display the image using PIL
                import PIL.Image
                image = PIL.Image.open(image_path)
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())

            except Exception as e:
                # Handle any errors that may occur while opening or displaying the image
                window["-RESULT-"].update(f"Error: {str(e)}")

window.close()

# increase the size of the font
# fix button container 
# have analyze button