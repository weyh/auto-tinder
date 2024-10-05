# Auto tinder

The goal of this project is to automate the tedious task of swiping. It should work with all dating apps on android that has a swipe feature.

## An apology

This code is not good. It all should have been a simple py script, but we cannot have nice things. 🙃

## Project structure

### Image preprocessor

Folder: `py-img-pre-proc`

The goal of this script is to copy the training data to the right place while compressing it.

In the input folder it expects files that are made by the android app. So the file name should start with "ok" or "x" and match `(ok|x)_[0-9]+.(jpg|png)` pattern. These files can also be in a zip just to group them.

The output folder is the folder where the image classifier tensorflow py script expects them, so `py-img-classification/cache/data`. There it will create an "ok" and a "x" folder.

### Image classifier

Folder: `py-img-classification`

The `main.py` file is responsible for training the TF model. It will output a TF Lite file. It expects the images to be in `cache/data` folder. (I just kind of went with the model params that worked for me.)

`predict_server.py` is a simple TCP server that does the prediction for the android app, because I was unable to get the andorid tflite interpreter to put out sane results. So we need to run this on something, like a pi 🙂.

### Auto swiper and screenshot taker app

Folder: `TinderScreenshotTaker`

Initially I wanted a separate app for swiping but I was too lazy so this monstrosity was born. Also while I was at it I wanted to try out kotlin so half of it is in kotlin for no good reason.
This app requires all the permissions under the sun, to get things done.

With the ❤️ and 🚫 overlay buttons you can take screenshots which are saved in the downloads folder. With the 🦾 button you can use the prediction functionality. If the auto swipe checkbox is checked and the ip and port are correctly set to the predict_server.py, it will start automatically swiping.

## Workflow

1. Take screenshots via the app or otherwise (or use other sources)
2. Copy the images to a storage space where the preprocessor can access it
3. Run the preprocessor
4. Train model (if it's not good enough go back to step 1)
5. Copy the model to a folder that the prediction server can access
6. Start the server
8. Set the ip and port, enable auto swipe if you want auto swiping to happen based on the response from the model
9. Use the 🦾 button to evaluate the current content of your screen

## Obtaining data

The best recourse is screenshots of the dating apps, but you can augment them with other data sets or scraped data from social media. I've started achieving usable results with around 5000 images. It's also important to get about equal amount of images from all dating apps because otherwise the model will have some biases.

## License
This project is licensed under the GNU AGPLv3 License - see the [GNU AGPLv3 License](LICENSE) file for details.