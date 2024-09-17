import numpy as np
from scipy.signal import butter, lfilter

class PCPS :
	def __init__ (self) :
		self.N = 150
		self.max_threshold = None
		self.increments = None
		self.predict_luminance = None

	def setThreshold (self, thresh) :
		'''Set the max threshold to determine if peaks exceed it'''
		self.max_threshold = thresh
		print(f'Set threshold {thresh}')

	def setIncrements(self, inc) :
		'''Set the pupil size at luminance increments'''

		self.increments = inc
		print(f'Set increments {inc}')

		# the fit_luminance_function() was calculating this poly1d for
		# every sample and it is the same result each time
		# only calculate it once and use the poly1d only to predict

		x = np.linspace(0, 1, len(inc), endpoint=True)
		y = inc
		coeffs = np.polyfit(x, y, 3)
		self.predict_luminance = np.poly1d(coeffs)

	def predictPupilSizeAtLuminance (self, luminance) :
		'''Return the predicted pupil size at luminance values'''
		return self.predict_luminance(luminance)

	def predictPupilSizes (self, luminances) :
		return np.array( [self.predictPupilSizeAtLuminance(l) for l in luminances] )

	def preprocessPupil(self, y):
		'''Preprocess pupil size by removing out of band values'''
		for k in range(0, len(y)):
			if (y[k] < 0.8) or (y[k] > 10):
				y[k] = 'nan'
			y_interpolated = np.interp(np.arange(len(y)), np.arange(len(y))[np.isnan(y) == False], y[np.isnan(y) == False])
		return y_interpolated

	def calculateWorkload (self, pupil_left, luminances) :
		'''Pass in two lists of floats which will be converted to numpy arrays'''
		pupil_left = np.array(pupil_left)
		predicted_pupil_left = self.predictPupilSizes(luminances)
		cleaned_pupil_left = self.preprocessPupil(pupil_left)
		luminance_removed_pupil_left = np.zeros((len(cleaned_pupil_left)))
		for i in range(0, len(cleaned_pupil_left)):
			luminance_removed_pupil_left[i] = cleaned_pupil_left[i] - predicted_pupil_left[i]
		y_padded = np.pad(luminance_removed_pupil_left, (self.N//2, self.N-1-self.N//2), mode='edge')
		predicted_pupil_left_moving_avg = np.convolve(y_padded, np.ones((self.N,))/self.N, mode='valid')
		power = predicted_pupil_left_moving_avg ** 2
		moving_avg_power = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
		peak_point = np.max(moving_avg_power)
		if peak_point > self.max_threshold :
			return 1
		else :
			return 0