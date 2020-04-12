from flask import Flask

app = Flask(__name__)


@app.route('/execute_ml_model',method=['GET','POST'])
def execute_ml_model():
    ## load pickle for model created

    # this request whatever new values have come they will predit the Y value using the model loaded from the picket
    return 'completed'

