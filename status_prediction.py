import sys
import json
import os
import face_recognition
from sklearn import neighbors
import _pickle as cPickle

database_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'database.clf')
clf = cPickle.load(open(database_file,"rb"))

def format_error(message):
    return ({'error': message})

def format_result(n, l, s):
    return ({'name': n, 'coordinates': l, 'score': s})


def predict(image):
    X_face_locations = face_recognition.face_locations(image)
    if len(X_face_locations) == 0:
        return format_error('No faces found')

    faces_encodings = face_recognition.face_encodings(
        image, known_face_locations=X_face_locations)
    closest_distances = clf.kneighbors(faces_encodings, n_neighbors=5)
    scores = [closest_distances[0][i][0] for i in range(len(X_face_locations))]
    predicitons = zip(clf.predict(faces_encodings), X_face_locations, scores)

    return ({'predictions': [format_result(n, l, s) if s <= 0.5 else format_result('unknown', l, s) for n, l, s in predicitons]})


if __name__ == "__main__":
    if sys.argv[1:]:
        image = sys.argv[1]
        face_rec = face_recognition.load_image_file(image)
        response = predict(face_rec)
        print(json.dumps(response))

    else:
        result = format_error('No file provided')
        print(json.dumps(result))
        sys.exit()
