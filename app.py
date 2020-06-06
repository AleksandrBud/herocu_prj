import h2o
import logging
import traceback
import time
from flask import Flask, request, jsonify
from process_data import process_record
from logging.handlers import RotatingFileHandler
from time import strftime


app = Flask(__name__)


h2o.init()
model_my = h2o.load_model('./models/GLM_model_python_1591360654781_1')
handler = RotatingFileHandler('error.log', maxBytes=500000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@app.route('/')
def index():
    return 'API Flask work!'


@app.route('/predict', methods=['POST'])
def predict():
    curr_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    ip_adr = request.headers.get('X-Forwarded For', request.remote_addr)
    logger.info(f'{curr_datatime} request from {ip_adr}: {request.json}')
    start_pred = time.time()

    json_input = request.json
    prc_date = process_record(json_input)
    id_v = json_input['ID']
    h2o_df = h2o.H2OFrame(prc_date)
    predict_val = model_my.predict(h2o_df)
    value = predict_val.as_data_frame()['predict'][0].astype(float)
    result = dict(ID=id_v, ClaimInd=value)

    end_predict = time.time()
    duration = round(end_predict - start_pred, 6)
    curr_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    logger.info(f'{curr_datatime} process time {duration} msec: {request.json}\n')

    return jsonify(result)


@app.errorhandler(Exception)
def exceptions(e):
    date_time = strftime('[%Y-%b-%d %H:%M:%S]')
    error_mes = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx SERVER ERROR\n%s',
                 date_time,
                 request.remote_addr,
                 request.method,
                 request.scheme,
                 request.full_path,
                 error_mes)
    return jsonify({'error': 'Server ERROR'}), 500


if __name__ == '__main__':
    app.run(debug=False)
