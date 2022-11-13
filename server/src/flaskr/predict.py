import re
import base64
from io import BytesIO

from flask import Blueprint, jsonify, request, current_app

from inference import forward


bp = Blueprint('predict', __name__, url_prefix='/predict')


@bp.route("/image", methods=['POST'])
def predict():
    data = request.json
    # getting the base64 file and removing metadata prefix
    base64_image = re.sub('^data:image/.+;base64,', '', request.json['image'])
    # converting base64 to image
    image = BytesIO(base64.b64decode(base64_image))

    class_name, certainty = forward(
        model=current_app.config['model'], image=image)

    certainty = certainty.item() * 100

    if class_name == "Normal":
        certainty = 100 - certainty

    return jsonify(
        {
            'image': data['filename'],
            'prediction': {
                'class': class_name.lower(),
                'certainty': certainty,
            }
        }
    )
