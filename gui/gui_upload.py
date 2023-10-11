import PySimpleGUI as sg
from PIL import Image
import io

layout = [
    [sg.Text('Select an image to upload:')],
    [sg.InputText(key='file_path', enable_events=True), sg.FileBrowse('Upload', key='upload_button')],
    [sg.Image(key='image')],
    [sg.Button('Exit')]
]

window = sg.Window('Image Upload and Display Example', layout, resizable=True)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'upload_button':
        file_path = values['file_path']
        if file_path:
            try:
                # Open the image using PIL
                image = Image.open(file_path)

                # Resize the image to fit in the window
                image.thumbnail((400, 400))

                # Convert the PIL image to bytes for PySimpleGUI
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                window['image'].update(data=img_bytes.getvalue())

            except Exception as e:
                sg.popup_error(f'Error: {str(e)}')

window.close()
