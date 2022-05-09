import io
from PIL import Image
import wrapper
import flask
import numpy as np

app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        if flask.request.files.get("img"):
            img = Image.open(io.BytesIO(flask.request.files["img"].read()))
            img = np.array(img)
            session = wrapper.open_sess("./yolox_v2.onnx")
            final_boxes, final_scores, final_cls_inds = wrapper.run(sess=session, img=img, visual=False)
            prediction = flask.jsonify({"final_boxes": final_boxes.tolist(), "final_scores": final_scores.tolist(), "final_cls_inds": final_cls_inds.tolist()})
        else:
            prediction = "ERROR no img"
    else:
        prediction = "ERROR not POST"
    return prediction

@app.route("/sanity")
def sanity():
    return "Hello World"

if __name__ == "__main__":
    app.run(host='0.0.0.0')