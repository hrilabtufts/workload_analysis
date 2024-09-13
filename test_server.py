from argparse import ArgumentParser
from os.path import exists, isfile
from PCPS import PCPS
import requests

parser = ArgumentParser(description='Script for testing pcps workload analysis server')
parser.add_argument('eye_tracking_raw', type=str, help='The Eye_Tracking_Raw.csv to process')
parser.add_argument('eye_tracking_events', type=str, help='The Eye_Tracking_Events.csv to process')
parser.add_argument('--threshold', '-t', type=float, default=-1.0, help='Threshold value (default: -1.0)')
parser.add_argument('--url', '-u', type=str, default='http://localhost:9995', help='URL of server to test')
parser.add_argument('--seconds', '-s', type=int, default=30, help='Period, in seconds, over which to average pupil size (default: 30)')
args = parser.parse_args()


if not exists(args.eye_tracking_raw) :
    print(f'ERROR: File {args.eye_tracking_raw} does not exist')
    exit(1)

if not isfile(args.eye_tracking_raw) :
    print(f'ERROR: {args.eye_tracking_raw} is not a file')
    exit(2)

if not exists(args.eye_tracking_events) :
    print(f'ERROR: File {args.eye_tracking_events} does not exist')
    exit(3)

if not isfile(args.eye_tracking_events) :
    print(f'ERROR: {args.eye_tracking_events} is not a file')
    exit(4)


threshobj = {'threshold': args.threshold}

x = requests.post(f'{args.url}/threshold', json = threshobj)

print(f'Set threshold {x.text}')

calibration_steps=[]
with open(args.eye_tracking_events, 'r') as file :
    for l in file:
        line = l.strip()
        cols = line.split(',')
        if cols[2].startswith('pupil_calibration_') :
            if '_started' in cols[2] :
                continue
            parts = cols[2].split('_')
            if parts[2] == '0' :
            	continue
            if parts[2] == 'ended' :
            	parts[2] = '270'
            calibration_step = {
                'time' : float(cols[0]),
                'step' : int(parts[2]) - 15
            }
            calibration_steps.append(calibration_step)

raw=[]
pupil_at_incrementation = []
with open(args.eye_tracking_raw, 'r') as file :
    for l in file:
        line = l.strip()
        if line.startswith('unityClientTimestamp,') :
            continue
        cols = line.split(',')
        raw.append([ float(cols[0]), float(cols[15]) ])

c = 0
for i in range(len(raw)) :
    cols = raw[i]
    if c == len(calibration_steps) :
        break
    if cols[0] >= calibration_steps[c]['time'] :
        pupil = cols[1]
        if pupil < 0 :
            j = 1
            while pupil < 0 :
                pupil = raw[i-j][1]
                j += 1
        pupil_at_incrementation.append(pupil)
        c+=1

pupil_at_incrementation = [str(i) for i in pupil_at_incrementation]
inc_str = ','.join(pupil_at_incrementation)

incobj = {'increments': inc_str}

x = requests.post(f'{args.url}/increments', json = incobj)

print(f'Set incrementations {x.text}')

with open(args.eye_tracking_raw, 'r') as file :
    start_time = -1

    pupil_left_arr = []
    luminance_arr = []

    for l in file:
        line = l.strip()

        #unityClientTimestamp,unityClientLocalTimestamp,deviceTimestamp,gazeOriginLeftX,gazeOriginLeftY,gazeOriginLeftZ,gazeOriginRightX,gazeOriginRightY,gazeOriginRightZ,gazeDirectionLeftX,gazeDirectionLeftY,gazeDirectionLeftZ,gazeDirectionRightX,gazeDirectionRightY,gazeDirectionRightZ,pupilDiameterLeft,pupilDiameterRight,eyeOpennessLeft,eyeOpennessRight,pupilPositionLeftX,pupilPositionLeftY,pupilPositionRightX,pupilPositionRightY,luminance
        if line.startswith('unityClientTimestamp,') :
            continue

        cols = line.split(',')

        if start_time < 0 :
            start_time = int(cols[2])

        time = int(cols[2])
        pupil_left = float(cols[15])
        luminance = float(cols[23])

        pupil_left_arr.append(pupil_left)
        luminance_arr.append(luminance)
        
        if time - start_time >= args.seconds * 1000 :
            pupil_str = ",".join([str(i) for i in pupil_left_arr])
            luminance_str = ",".join([str(i) for i in luminance_arr])
            workloadobj = {'pupil' : pupil_str, 'luminance' : luminance_str}
            x = requests.post(f'{args.url}/workload', json = workloadobj)
            workload = int(x.text)
            print(f'{cols[2]},{workload}')

            pupil_left_arr = []
            luminance_arr = []
            start_time = -1