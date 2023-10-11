import PySimpleGUI as sg
import io

# Define the home layout
home_layout = [
    [sg.Text("Welcome to the Rice Leaf Disease Classification")],
    [sg.Button("Start")],
]

# Define the second layout
second_layout = [
    [sg.Text("Upload an image:")],
    [sg.InputText(key="-IMAGE_PATH-"), sg.FileBrowse(file_types=(("Images", "*.png *.jpg *.jpeg *.gif"),))],
    [sg.Button("Upload")],
    [sg.Image(key="-IMAGE-", size=(300, 300))],
    [sg.Text("Analysis Result:")],
    [sg.Text("", size=(30, 1), key="-RESULT-")],
]

# Create the main window
layout = [
    [sg.Column(home_layout, key="-HOME-", visible=True)],
    [sg.Column(second_layout, key="-SECOND-", visible=False)],
]

window = sg.Window("Rice Leaf Disease Classification", layout, finalize=True, size=(750, 750))

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
