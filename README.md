# Facial Recognition

## Setup

`pip3 install sklearn Pillow face_recognition` to get all required libraries for this project.

## Training

Create a folder for each person using their name in lowercase and underscores for spaces e.g. `jack_doe` `john_doe`.
Fill the folder with images of just that person and make sure no other faces are in the image and there is more than 1 image.

run `python3 ./status_training.py`, once complete the `database.clf` will be updated which will be used for the predictions.

## Predictions

To run a facial recognition on an image all you have to do is run `python3 ./status_prediction.py ./path/to/image.jpg` and the
script will return the results of anyone it picks up in the image, Results will be returned as JSON.

```json
[
  {
    "name": "unknown",
    "coordinates": [982, 957, 1025, 914],
    "score": 0.6941232641733543
  },
  {
    "name": "john_doe",
    "coordinates": [660, 1447, 703, 1404],
    "score": 0.36921533611616386
  },
  {
    "name": "jack_doe",
    "coordinates": [391, 957, 434, 914],
    "score": 0.36302260895814914
  },
  {
    "name": "unknown",
    "coordinates": [430, 1567, 473, 1524],
    "score": 0.6223868744925297
  }
]
```

You can run the script on the test data provided like so `python3 ./status_prediction.py test_data/faces.jpg`

if no faces are detected an error will be returned `{"error": "No faces found"}`
