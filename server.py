from argparse import ArgumentParser
from PCPS import PCPS
from flask import Flask, render_template, request, redirect, session
import json

parser = ArgumentParser(description='Server for hosting the pcps workload analysis')
parser.add_argument('--port', '-p', type=int, default=9995, help='Port for server (default: 9995)')
parser.add_argument('--threshold', '-t', type=float, default=-1.0, help='Threshold value (default: -1.0)')
args = parser.parse_args()

pcps = PCPS()
pcps.setThreshold(args.threshold)

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home () :
    return 'OK'

@app.route('/threshold', methods = ['POST'])
def threshold () :
	data = request.get_json()
	thresh = float(data['threshold'])
	pcps.setThreshold(thresh)
	return 'OK'

@app.route('/increments', methods = ['POST'])
def increments () :
	data = request.get_json()
	inc =  [float(i) for i in data['increments'].split(',')]
	pcps.setIncrements(inc)
	return 'OK'

@app.route('/workload', methods = ['POST'])
def workload ():
	data = request.get_json()
	pupil = [float(i) for i in data['pupil'].split(',')]
	luminance = [float(i) for i in data['luminance'].split(',')]
	return str(pcps.calculateWorkload(pupil, luminance))

if __name__ == '__main__' :
    app.run(host='0.0.0.0', port=args.port, debug=False)